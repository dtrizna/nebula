import os
import logging
import numpy as np
from tqdm import tqdm
from time import time
from typing import Union, List
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from .lit_utils import LitTrainerWrapper, PyTorchLightningModel


class PyTorchLightningModelLM(PyTorchLightningModel):
    def __init__(self, *args, **kwargs):
        super().__init__(loss = CrossEntropyLoss(), *args, **kwargs)
        assert hasattr(self.model, "pretrain"), "This model does not have a 'pretrain' method."
        self.forward = self.model.pretrain


class MaskedLanguageModelTrainer(LitTrainerWrapper):
    def __init__(
            self,
            vocab: dict,
            pretrain_epochs: int = 3,
            # whether to remask sequence after nr of remask_epochs
            remask_epochs: int = False,
            mask_probability: float = 0.15,
            # mask the token with distribution (80% mask, 10% random, 10% same)
            # same as Devlin et al (https://arxiv.org/abs/1810.04805)
            mask_distribution: list = [0.8, 0.1],
            # how to construct the target for the masked tokens
            # 1 if onehot, sum of maskings if count
            masked_target_type: str = "onehot",
            dump_model_every_epoch: bool = False,
            *args,
            **kwargs
    ):
        super().__init__(skip_trainer_init=True, monitor_metric="train_loss", monitor_mode="min", *args, **kwargs)
        self.__name__ = "MaskedLanguageModel"
        
        assert "<pad>" in vocab, "Vocabulary must contain '<pad>' token"
        assert "<mask>" in vocab, "Vocabulary must contain '<mask>' token"
        self.vocab = vocab
        self.pretrain_epochs = pretrain_epochs
        
        if remask_epochs:
            assert isinstance(remask_epochs, int), "remask_epochs must be an integer"
        self.remask_epochs = remask_epochs

        assert len(mask_distribution) == 2, "mask_distribution must be a list of length 2"
        self.mask_distribution = mask_distribution
        
        self.mask_probability = mask_probability
        
        assert masked_target_type in ["onehot", "count"], "masked_target_type must be either 'onehot' or 'count'"
        self.masked_target_type = masked_target_type
        
        self.dump_model_every_epoch = dump_model_every_epoch
        

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


    def setup_lit_language_model(self):
        self.lit_model = PyTorchLightningModelLM(
                model=self.pytorch_model,
                learning_rate=self.learning_rate,
                scheduler=self.scheduler,
                scheduler_step_budget=self.scheduler_budget,
            )

    def pretrain(self, x_unlabeled: np.ndarray, epochs: int = None):
        if epochs is not None:
            self.pretrain_epochs = epochs

        # MODEL SETUP
        if self.lit_model is None:
            self.setup_lit_language_model()

        # DATA SETUP
        logging.warning(' [*] Masking of sequences...')
        x_masked, y_masked = self.mask_seq_arr(x_unlabeled)
        dataloader = self.create_dataloader(x_masked, y_masked, shuffle=True)
        
        # TRAINER SETUP AND TRAINING
        self.epochs = 1
        self.setup_trainer()
        for epoch in range(self.pretrain_epochs):
            if self.remask_epochs and epoch % self.remask_epochs == 0 and epoch > 0:
                logging.warning(f' [*] Re-masking sequences...')
                x_masked, y_masked = self.mask_seq_arr(x_unlabeled)
                dataloader = self.create_dataloader(x_masked, y_masked, shuffle=True)

            self.trainer.fit(self.lit_model, dataloader)

            # Modify the Trainer's fit_loop max_epochs to ensure .fit() can be called again
            if epoch < self.pretrain_epochs - 1:
                self.trainer.fit_loop.max_epochs = self.trainer.max_epochs + 1

            if self.dump_model_every_epoch:
                self.save_torch_model(model_file=os.path.join(self.log_folder, f"pretrained_epoch_{epoch}.torch"))

        # NOTE: lit weird behavior: no checkpoint files are saved if train_loss 
        # is specified as monitor metric in ModelCheckpoint, therefore, saving torch model manually.
        self.save_torch_model(model_file=os.path.join(self.log_folder, "pretrained_final.torch"))


class AutoRegressiveModelTrainer(LitTrainerWrapper):
    def __init__(
            self,
            vocab: dict,
            pretrain_epochs: int = 3,
            block_size: int = 256,
            dump_model_every_epoch: bool = False,
            random_offsets: int = False,
            *args,
            **kwargs
    ):
        super().__init__(skip_trainer_init=True, *args, **kwargs)
        self.__name__ = "AutoRegressiveModel"

        assert "<pad>" in vocab, "Vocabulary must contain '<pad>' token"
        assert "<mask>" in vocab, "Vocabulary must contain '<mask>' token"
        self.vocab = vocab

        self.pretrain_epochs = pretrain_epochs
        self.block_size = block_size
        self.dump_model_every_epoch = dump_model_every_epoch
        self.random_offsets = random_offsets


    def pretrain(        
        self,
        x_unlabeled: np.ndarray,
    ):        
        if self.random_offsets:
            assert isinstance(self.random_offsets, int), "random_offsets must be an integer"
            ix = np.random.randint(x_unlabeled.shape[0]-self.block_size, size=(self.random_offsets,))
            x = np.stack([x_unlabeled[i:i+self.block_size] for i in ix])
            y = np.stack([x_unlabeled[i+1:i+self.block_size+1] for i in ix])
        else: # sequential
            x = np.stack([x_unlabeled[i:i+self.block_size] for i in range(x_unlabeled.shape[0]-self.block_size)])
            y = np.stack([x_unlabeled[i+1:i+self.block_size+1] for i in range(x_unlabeled.shape[0]-self.block_size)])

        # TBD
        raise NotImplementedError


class SelfSupervisedLearningEvalFramework:
    def __init__(
            self,
            # pretrainer: Union[MaskedLanguageModelTrainer, AutoRegressiveModelTrainer],
            pretrainer: MaskedLanguageModelTrainer,
            downstream_trainer: LitTrainerWrapper,
            training_types: List[str] = ['pretrained', 'non_pretrained', 'full_data'],
            # eval details
            unlabeled_data_ratio: float = 0.8,
            n_splits: int = 5,
            # logging details
            log_folder: str = None,
            dump_data_splits: bool = True,
            downsample_unlabeled_data: bool = False,
            false_positive_rates: List[float] = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            random_state: int = None,
    ):
        self.pretrainer = pretrainer
        self.downstream_trainer = downstream_trainer

        self.training_types = training_types
        self.unlabeled_data_ratio = unlabeled_data_ratio
        self.n_splits = n_splits

        tempfolder = os.path.join(os.getcwd(), f"out_self_supervised_eval_{int(time())}")
        self.log_folder = log_folder if log_folder is not None else tempfolder
        os.makedirs(self.log_folder, exist_ok=True)

        self.pretrainer.log_folder = os.path.join(self.log_folder, self.pretrainer.log_folder)
        self.init_pretrain_log_folder = self.pretrainer.log_folder
        
        self.downstream_trainer.log_folder = os.path.join(self.log_folder, self.downstream_trainer.log_folder)
        self.init_downstream_log_folder = self.downstream_trainer.log_folder
        
        self.init_pretrain_model_weights = self.pretrainer.pytorch_model.state_dict()
        self.init_downstream_model_weights = self.downstream_trainer.pytorch_model.state_dict()        

        self.dump_data_splits = dump_data_splits
        self.downsample_unlabeled_data = downsample_unlabeled_data
        if self.downsample_unlabeled_data:
            assert isinstance(downsample_unlabeled_data, float) and 0 < downsample_unlabeled_data < 1
        self.false_positive_rates = false_positive_rates
        self.random_state = random_state


    def _dump_data_splits(self):
        split_data_file = f"dataset_splits_{self.timestamp}.npz"
        np.savez_compressed(
            os.path.join(self.log_folder, split_data_file),
            unlabeled_data=self.unlabeled_data,
            labeled_x=self.labeled_x,
            labeled_y=self.labeled_y
        )
        print(f"[!] Saved dataset splits to {split_data_file}")


    def run_one_split(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_val: np.ndarray = None,
            y_val: np.ndarray = None,
    ):
        self.timestamp = int(time())

        # split x and y into train and validation sets
        self.unlabeled_data, self.labeled_x, _, self.labeled_y = train_test_split(
            x_train, y_train, train_size=self.unlabeled_data_ratio, random_state=self.random_state
        )
        
        if self.downsample_unlabeled_data:
            # sample N random samples from unlabeled data which is numpy array
            unlabeled_size = self.unlabeled_data.shape[0]
            indices = np.random.choice(unlabeled_size, int(self.downsample_unlabeled_data*unlabeled_size), replace=False)
            self.unlabeled_data = self.unlabeled_data[indices].copy()
        
        if self.dump_data_splits:
            self._dump_data_splits()

        print("[!] Pre-training model...")
        self.pretrainer.log_folder = self.init_pretrain_log_folder + "_" + str(self.timestamp)
        
        # reset model weights
        self.pretrainer.pytorch_model.load_state_dict(self.init_pretrain_model_weights)
        self.pretrainer.setup_trainer()
        self.pretrainer.setup_lit_language_model()
        self.pretrainer.pretrain(self.unlabeled_data)

        self.train_loader = self.downstream_trainer.create_dataloader(self.labeled_x, self.labeled_y, shuffle=True)
        if "full_data" in self.training_types:
            self.full_train_loader = self.downstream_trainer.create_dataloader(x_train, y_train, shuffle=True)
        
        if x_val is not None:
            self.val_loader = self.downstream_trainer.create_dataloader(x_val, y_val, shuffle=False)
        else:
            self.val_loader = None
        
        for training_type in self.training_types:
            print(f"[!] Fine-tuning of '{training_type}' model on downstream task...")
            self.downstream_trainer.name = training_type
            self.downstream_trainer.log_folder = self.init_downstream_log_folder + "_" + training_type + "_" + str(self.timestamp)

            if training_type == "pretrained":
                self.downstream_trainer.pytorch_model.load_state_dict(self.pretrainer.pytorch_model.state_dict())
            else:
                self.downstream_trainer.pytorch_model.load_state_dict(self.init_downstream_model_weights)
            
            self.downstream_trainer.setup_trainer()
            self.downstream_trainer.setup_lit_model()
            
            if training_type == "full_data":
                self.downstream_trainer.train_lit_model(self.full_train_loader, self.val_loader)
            else:
                self.downstream_trainer.train_lit_model(self.train_loader, self.val_loader)


    def run_splits(self, *args, **kwargs):
        for i in range(self.n_splits):
            self.random_state += i # to get different splits
            self.pretrainer.random_state = self.random_state
            self.downstream_trainer.random_state = self.random_state
            print(f'[!] Running pre-training split {i+1}/{self.n_splits}')
            self.run_one_split(*args, **kwargs)
