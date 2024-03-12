import os
import torch
import logging
import numpy as np
from tqdm import tqdm
from time import time
from typing import List, Optional, Union
from collections import OrderedDict
from copy import deepcopy
from torch import load
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .lit_utils import LitTrainerWrapper, PyTorchLightningModelLM
from .misc import clear_cuda_cache


class LanguageModelTrainer(LitTrainerWrapper):
    def __init__(
            self,
            vocab: dict,
            pretrain_epochs: int = 3,
            # whether to redo language model sampling after nr of remask epochs
            rebuild_dataloader_every_n_epochs: int = False,
            dump_model_every_epoch: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(skip_trainer_init=True, monitor_metric="train_loss", monitor_mode="min", *args, **kwargs)
        self.__name__ = "LanguageModelTrainer"
        
        assert "<pad>" in vocab, "Vocabulary must contain '<pad>' token"
        assert "<mask>" in vocab, "Vocabulary must contain '<mask>' token"
        self.vocab = vocab
        self.pretrain_epochs = pretrain_epochs
        self.dump_model_every_epoch = dump_model_every_epoch
        
        if rebuild_dataloader_every_n_epochs:
            assert isinstance(rebuild_dataloader_every_n_epochs, int),\
                "rebuild_dataloader_every_n_epochs must be an integer"
        self.rebuild_dataloader_every_n_epochs = rebuild_dataloader_every_n_epochs


    def setup_lit_language_model(self):
        if self.scheduler is not None and self.scheduler_budget is None:
            self.calculate_scheduler_step_budget(max_epochs=self.pretrain_epochs)

        self.lit_model = PyTorchLightningModelLM(
                model=self.pytorch_model,
                learning_rate=self.learning_rate,
                scheduler=self.scheduler,
                scheduler_step_budget=self.scheduler_budget,
            )


    def create_lm_dataloader(self, x_unlabeled: np.ndarray, shuffle: bool = True) -> DataLoader:
        raise NotImplementedError("create_lm_dataloader() must be implemented in subclass")


    def pretrain(self, x_unlabeled: np.ndarray, epochs: int = None):
        if epochs is not None:
            self.pretrain_epochs = epochs
        # DATA SETUP
        logging.warning('[*] Building language model dataloader...')
        self.train_loader = self.create_lm_dataloader(x_unlabeled)

        # MODEL SETUP
        if self.lit_model is None:
            self.setup_lit_language_model()
        
        # TRAINER SETUP AND TRAINING
        if not self.rebuild_dataloader_every_n_epochs and not self.dump_model_every_epoch:
            self.epochs = self.pretrain_epochs
            self.setup_trainer()
            self.trainer.fit(self.lit_model, self.train_loader)
        else:
            # need to stop training in between to peform extra logic: saving or remasking
            # TODO: this doesn't work with 'scheduler', since L.Trainer
            # calls configure_optimizers() with every .fit()
            # need to rewrite on_epoch_end() in Trainer or something
            # for now avoiding using remask_every_n_epochs and dump_model_every_epoch w/ scheduler    
            loop_length = self.rebuild_dataloader_every_n_epochs
            total_loops = self.pretrain_epochs // loop_length
            if self.dump_model_every_epoch:
                loop_length = 1
                total_loops = self.pretrain_epochs
            self.epochs = loop_length
            self.setup_trainer()
            for loop in range(total_loops):
                if loop > 0 and (not self.dump_model_every_epoch or loop % self.rebuild_dataloader_every_n_epochs == 0):
                    logging.warning(f'[*] Re-building language model dataloader...')
                    self.train_loader = self.create_lm_dataloader(x_unlabeled)

                self.trainer.fit(self.lit_model, self.train_loader)

                # NOTE: here we modify the Trainer's fit_loop max_epochs to ensure .fit() can be called again
                if loop < total_loops - 1:
                    self.trainer.fit_loop.max_epochs = self.trainer.max_epochs + loop_length

                if self.dump_model_every_epoch:
                    self.save_torch_model(os.path.join(self.log_folder, f"pretrained_epoch_{loop}.torch"))
        
        # NOTE: lit weird behavior: no checkpoint files are saved if train_loss 
        # is specified as monitor metric in ModelCheckpoint, therefore, saving torch model manually.
        self.save_torch_model(os.path.join(self.log_folder, "pretrained_final.torch"))


class MaskedLanguageModelTrainer(LanguageModelTrainer):
    def __init__(
            self,
            mask_probability: float = 0.15,
            # mask the token with distribution (80% mask, 10% random, 10% same)
            # same as Devlin et al (https://arxiv.org/abs/1810.04805)
            mask_distribution: list = [0.8, 0.1],
            # how to construct the target for the masked tokens
            # 1 if onehot, sum of maskings if count
            masked_target_type: str = "onehot",
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__name__ = "MaskedLanguageModelTrainer"
        
        assert len(mask_distribution) == 2, "mask_distribution must be a list of length 2"
        self.mask_distribution = mask_distribution
        
        self.mask_probability = mask_probability
        
        assert masked_target_type in ["onehot", "count"], "masked_target_type must be either 'onehot' or 'count'"
        self.masked_target_type = masked_target_type


    def mask_seq(self, sequence: np.ndarray):
        """
        Mask a sequence with a given probability.
        
        Masks 80% of tokens as a mask token, 10% as a random token, and 10% stays the same.
        Also returns a sequence with 1s in masked positions, and a sequence with 1s in the original positions.

        Parameters:
        - sequence: one dimensional numpy array (aka vector) with the input sequence
        - mask_probability: the probability of masking a token (default: 0.15)
        - vocab: the vocabulary used during encoding
        - random_state: the random state to use for reproducibility (default: None)
        - token_id_type: the type of replaces element vocabulary, either: "onehot" or "count"

        Returns:
        - a tuple with the masked sequence, the mask, and the input size sequence
        """
        vocabSize=len(self.vocab)
        masked_token_idxs = np.zeros(vocabSize, dtype=np.int32)

        # limit sequence till first padding token to avoid masking padding
        if self.vocab["<pad>"] in sequence:
            masked_sequence = sequence[:np.where(sequence == self.vocab["<pad>"])[0][0]].copy()
        else:
            masked_sequence = sequence.copy()

        # find out which tokens to mask and loop over    
        maskIdxs = np.random.uniform(size=masked_sequence.shape) < self.mask_probability
        for idx in np.where(maskIdxs)[0]:
            # prepare array of vocabSize that specifies which tokens were masked
            tokenId = masked_sequence[idx]
            if self.masked_target_type == "count":
                masked_token_idxs[tokenId] += 1
            else:
                masked_token_idxs[tokenId] = 1

            # actually mask the token with distribution (80% mask, 10% random, 10% same)
            # same as Devlin et al (https://arxiv.org/abs/1810.04805)
            sample = np.random.sample()
            if sample < self.mask_distribution[0]:
                masked_sequence[idx] = self.vocab["<mask>"]
            elif sample < (self.mask_distribution[0] + self.mask_distribution[1]):
                masked_sequence[idx] = np.random.randint(0, vocabSize)
            else:
                masked_sequence[idx] = sequence[idx]

        # pad masked sequence to be the same length as original sequence
        origSequenceLength = sequence.squeeze().shape[0]
        padWidth = origSequenceLength - masked_sequence.shape[0]
        masked_sequence = np.pad(
            array=masked_sequence,
            pad_width=(0, padWidth),
            mode='constant',
            constant_values=self.vocab["<pad>"]
        )

        return masked_sequence, masked_token_idxs


    def mask_seq_arr(self, seq_array: np.ndarray):
        seq_masked = []
        target = []
        for seq in tqdm(seq_array):
            masked_local, target_local = self.mask_seq(seq)
            seq_masked.append(masked_local)
            target.append(target_local)
        # U_masked shape: (n_sequences, max_len)
        # U_target shape: (n_sequences, vocab_size)
        seq_masked = np.vstack(seq_masked)
        target = np.vstack(target)
        return seq_masked, target


    def create_lm_dataloader(self, x_unlabeled: np.ndarray, shuffle: bool = True) -> DataLoader:
        x_masked, y_masked = self.mask_seq_arr(x_unlabeled)
        return self.create_dataloader(x_masked, y_masked, shuffle=shuffle)



class AutoRegressiveModelTrainer(LanguageModelTrainer):
    def __init__(
            self,
            context_len: int = 256,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__name__ = "AutoRegressiveModelTrainer"
        self.context_len = context_len


    def get_contexts(self, x_unlabeled: np.ndarray, size: int) -> np.ndarray:
        # NOTE: From: https://github.com/karpathy/nanoGPT/blob/master/train.py#L118
        # but sample nr of random_sample_idx from 0th dim, and then sample block_size from 1st dim
        # since x_unlabeled.shape is (number_of_samples, sequence_length), not (sequence_length, )
        max_len = x_unlabeled.shape[1]
        random_sample_idx = torch.randint(x_unlabeled.shape[0], size=(size,))
        random_start_idx = torch.randint(max_len-self.context_len, size=(size,))
        x = torch.stack([torch.from_numpy((x_unlabeled[i, j:j+self.context_len]).astype(np.int64)) for i, j in zip(random_sample_idx, random_start_idx)])
        y = torch.stack([torch.from_numpy((x_unlabeled[i, j+1:j+1+self.context_len]).astype(np.int64)) for i, j in zip(random_sample_idx, random_start_idx)])
        return x, y


    def create_lm_dataloader(self, x_unlabeled: np.ndarray, shuffle: bool = True) -> DataLoader:
        x, y = self.get_contexts(x_unlabeled, size=x_unlabeled.shape[0])
        # TODO: y shape is torch.Size([nr_samples, 256])
        # in case of masking it is torch.Size([nr_samples, vocab_size]) with masked elements marked as 1
        # need to think how to update the model to give existing value to the Transformer
        # but may be no updates needed, since last layer in Karpathy's GPT is the same as in our LM
        # might be of value: https://github.com/karpathy/nanoGPT/blob/master/model.py#L187C13-L187C104
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        loader = self.create_dataloader(x, y, shuffle=shuffle)
        return loader


class SelfSupervisedLearningEvalFramework:
    def __init__(
            self,
            pretrainer: Union[MaskedLanguageModelTrainer, AutoRegressiveModelTrainer],
            downstream_trainer: LitTrainerWrapper,
            training_types: List[str] = ['pretrained', 'non_pretrained', 'full_data'],
            # eval details
            unlabeled_data_ratio: float = 0.8,
            supervised_data_ratio: float = None,
            n_splits: int = 5,
            dump_data_splits: bool = True,
            # logging details
            log_folder: str = None,
            false_positive_rates: List[float] = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            random_state: int = None,
            pretrained_model_path: str = None,
    ):
        self.pretrainer = pretrainer
        self.downstream_trainer = downstream_trainer

        assert all(training_type in ['pretrained', 'non_pretrained', 'full_data'] for training_type in training_types), \
            "training_types should contain only 'pretrained', 'non_pretrained', or 'full_data' elements."
        self.training_types = training_types

        # supervised and unsupervised ratio checks
        if supervised_data_ratio is not None and unlabeled_data_ratio is not None:
            assert 0 < supervised_data_ratio + unlabeled_data_ratio <= 1,\
                "The sum of supervised_data_ratio and unlabeled_data_ratio cannot be greater than 1"
        elif supervised_data_ratio is None and unlabeled_data_ratio is not None:
            assert 0 <= unlabeled_data_ratio <= 1, "unlabeled_data_ratio should be between 0 and 1"
            supervised_data_ratio = 1 - unlabeled_data_ratio
        else:
            assert 0 <= supervised_data_ratio <= 1, "supervised_data_ratio should be between 0 and 1"
            unlabeled_data_ratio = 1 - supervised_data_ratio
        
        self.supervised_data_ratio = supervised_data_ratio
        self.unlabeled_data_ratio = unlabeled_data_ratio

        self.n_splits = n_splits
        self.dump_data_splits = dump_data_splits

        tempfolder = os.path.join(os.getcwd(), f"out_self_supervised_eval_{int(time())}")
        self.log_folder = log_folder if log_folder is not None else tempfolder
        os.makedirs(self.log_folder, exist_ok=True)

        self.pretrainer.log_folder = os.path.join(self.log_folder, self.pretrainer.log_folder)
        self.init_pretrain_log_folder = self.pretrainer.log_folder
        
        self.downstream_trainer.log_folder = os.path.join(self.log_folder, self.downstream_trainer.log_folder)
        self.init_downstream_log_folder = self.downstream_trainer.log_folder
        
        # NOTE: if not deepcopy, init_weights are overwritten with load_state_dict()
        self.init_pretrain_model_weights = deepcopy(self.pretrainer.pytorch_model.state_dict())
        self.init_downstream_model_weights = deepcopy(self.downstream_trainer.pytorch_model.state_dict())

        self.false_positive_rates = false_positive_rates
        self.random_state = random_state
        self.pretrained_model_path = pretrained_model_path


    def _dump_data_splits(self):
        split_data_file = f"dataset_splits_{self.timestamp}.npz"
        np.savez_compressed(
            os.path.join(self.log_folder, split_data_file),
            unlabeled_data=self.unlabeled_data,
            labeled_x=self.labeled_x,
            labeled_y=self.labeled_y
        )
        logging.warning(f"[!] Saved dataset splits to {split_data_file}")


    @staticmethod
    def _transfer_pretrained_weights(
        pretrained_state_dict: OrderedDict,
        downstream_state_dict: OrderedDict
    ) -> OrderedDict:
        """
        Transfer pretrained weights from a pretrained state dict to a downstream dict.
        """

        new_state_dict = deepcopy(downstream_state_dict)
        for name in downstream_state_dict:
            if name in pretrained_state_dict:
                new_state_dict[name] = deepcopy(pretrained_state_dict[name])

        return new_state_dict


    def _train_downstream_model(self, training_type: str) -> None:

        self.downstream_trainer.log_folder = self.init_downstream_log_folder + "_" + training_type + "_" + str(self.timestamp)
        final_model_file = os.path.join(self.downstream_trainer.log_folder, f"{training_type}_final.torch")
        if os.path.exists(final_model_file):
            logging.warning(f"[!] Downstream model already exists at '{final_model_file}'")
            return
        
        # reset params that should be re-initialized
        self.downstream_trainer.lit_model = None
        self.downstream_trainer.scheduler_budget = None
        self.downstream_trainer.name = training_type
        
        if training_type == "pretrained":
            self.downstream_trainer.pytorch_model.load_state_dict(self.pretrained_weights)
        else:
            self.downstream_trainer.pytorch_model.load_state_dict(self.init_downstream_model_weights)
        
        self.downstream_trainer.setup_trainer()

        if training_type == "full_data":
            self.downstream_trainer.train_loader = self.full_train_loader
            self.downstream_trainer.setup_lit_model()
            self.downstream_trainer.train_lit_model(self.full_train_loader, self.val_loader)
        else:
            self.downstream_trainer.train_loader = self.train_loader
            self.downstream_trainer.setup_lit_model()
            self.downstream_trainer.train_lit_model(self.train_loader, self.val_loader)
        self.downstream_trainer.save_torch_model(final_model_file)
        clear_cuda_cache()


    def run_one_split(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            idx: Optional[str] = None
    ):
        self.timestamp = int(time()) if idx is None else idx

        # split x and y into train and validation sets
        if os.path.exists(os.path.join(self.log_folder, f"dataset_splits_{self.timestamp}.npz")):
            logging.warning(f"[!] Loading dataset splits from 'dataset_splits_{self.timestamp}.npz'")
            splits = np.load(os.path.join(self.log_folder, f"dataset_splits_{self.timestamp}.npz"), allow_pickle=True)
            self.unlabeled_data = splits["unlabeled_data"]
            self.labeled_x = splits["labeled_x"]
            self.labeled_y = splits["labeled_y"]
        else:
            self.unlabeled_data, self.labeled_x, _, self.labeled_y = train_test_split(
                x_train, y_train,
                test_size=self.supervised_data_ratio,
                random_state=self.random_state
            )
        
        if self.supervised_data_ratio + self.unlabeled_data_ratio < 1:
            msg = f"[*] supervised ratio ({self.supervised_data_ratio}) + unlabeled ratio ({self.unlabeled_data_ratio}) < 1: downsampling unlabeled data..."
            logging.warning(msg)

            # sample a portion of unlabeled data which is numpy array
            size_of_dataset = x_train.shape[0]
            unlabeled_size = self.unlabeled_data.shape[0]
            nr_to_subsample = int(self.unlabeled_data_ratio*size_of_dataset)

            np.random.seed(self.random_state)
            indices = np.random.choice(unlabeled_size, nr_to_subsample)

            self.unlabeled_data = self.unlabeled_data[indices].copy()
            logging.warning(f"[!] Downsampled from {unlabeled_size} to {nr_to_subsample} unlabeled entries!")
        
        if self.dump_data_splits:
            self._dump_data_splits()
        
        # checking if pretrained model is provided or if already trained exists
        self.pretrainer.log_folder = self.init_pretrain_log_folder + "_" + str(self.timestamp)
        if self.pretrained_model_path is None:
            self.pretrained_model_path = os.path.join(self.pretrainer.log_folder, "pretrained_final.torch")
        if os.path.exists(self.pretrained_model_path):
            logging.warning(f"[!] Loading pretrained model from: '{self.pretrained_model_path}'")
            pretrained_model = load(self.pretrained_model_path)
            pretrained_model_state_dict = pretrained_model.state_dict()
        else:
            logging.warning(f"[!] Pre-training '{self.pretrainer.name}' model...")
            # reset model weights -- needed for multiple splits
            self.pretrainer.pytorch_model.load_state_dict(self.init_pretrain_model_weights)
            self.pretrainer.pretrain(self.unlabeled_data)
            pretrained_model_state_dict = deepcopy(self.pretrainer.pytorch_model.state_dict())
            clear_cuda_cache()

        self.pretrained_weights = self._transfer_pretrained_weights(
            pretrained_model_state_dict,
            self.init_downstream_model_weights
        )
        
        self.train_loader = self.downstream_trainer.create_dataloader(self.labeled_x, self.labeled_y, shuffle=True)
        if "full_data" in self.training_types:
            self.full_train_loader = self.downstream_trainer.create_dataloader(x_train, y_train, shuffle=True)
        
        if x_val is not None:
            self.val_loader = self.downstream_trainer.create_dataloader(x_val, y_val, shuffle=False)
        else:
            self.val_loader = None
        
        for training_type in self.training_types:
            logging.warning(f"[!] Fine-tuning of '{training_type}' model on downstream task...")
            self._train_downstream_model(training_type)


    def run_splits(self, x_train, y_train, x_val, y_val, previous_run_idxs: Optional[List] = None):
        if previous_run_idxs is not None:
            for idx in previous_run_idxs:
                self.run_one_split(x_train, y_train, x_val, y_val, idx=idx)
            return

        for i in range(self.n_splits):
            self.random_state += i # to get different splits
            self.pretrainer.random_state = self.random_state
            self.downstream_trainer.random_state = self.random_state
            logging.warning(f"[!] Running '{self.pretrainer.name}' pre-training split {i+1}/{self.n_splits}")
            self.run_one_split(x_train, y_train, x_val, y_val, idx=None)
