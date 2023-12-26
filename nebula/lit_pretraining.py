import os
import logging
import numpy as np
from tqdm import tqdm
from time import time
from typing import Union, List
from sklearn.model_selection import train_test_split

from torch.nn import CrossEntropyLoss

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
            # downsteam_epochs: int = 2, # TODO: do we care about it here?
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
        super().__init__(skip_trainer_init=True, *args, **kwargs)
        self.__name__ = "MaskedLanguageModel"
        
        assert "<pad>" in vocab, "Vocabulary must contain '<pad>' token"
        assert "<mask>" in vocab, "Vocabulary must contain '<mask>' token"
        self.vocab = vocab

        self.pretrain_epochs = pretrain_epochs
        # self.downsteam_epochs = downsteam_epochs

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


    def pretrain(self, x_unlabeled: np.ndarray, epochs: int = None):
        if epochs is not None:
            self.pretrain_epochs = epochs

        # MODEL SETUP
        if self.lit_model is None:
            self.lit_model = PyTorchLightningModelLM(
                model=self.pytorch_model,
                learning_rate=self.learning_rate,
                scheduler=self.scheduler,
                scheduler_step_budget=self.scheduler_budget,
            )

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
        # is specified as monitor metric in ModelCheckpoint, therefore, saving torch model.
        self.save_torch_model(model_file=os.path.join(self.log_folder, "pretrained_final.torch"))

        self.lit_model.forward = self.lit_model.model.forward


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


class SelfSupervisedPretraining:
    def __init__(
            self,
            lm_pretrainer: Union[MaskedLanguageModelTrainer, AutoRegressiveModelTrainer],
            downstream_trainer: LitTrainerWrapper,
            training_types: List[str] = ['pretrained', 'non_pretrained', 'full_data'],
            # pre-train details
            unlabeled_data_ratio: float = 0.8,
            pretrain_epochs: int = 5,
            downstream_epochs: int = 2,
            # logging details
            output_dir: str = None,
            dump_data_splits: bool = True,
            downsample_unlabeled_data: bool = False,
            false_positive_rates: List[float] = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
            random_state: int = None,
    ):
        self.pretrainer = lm_pretrainer
        self.downstread_trainer = downstream_trainer
        self.training_types = training_types
        self.unlabeled_data_ratio = unlabeled_data_ratio
        self.pretrain_pochs = pretrain_epochs
        self.downstream_epochs = downstream_epochs

        self.output_dir = output_dir
        self.dump_data_splits = dump_data_splits
        self.downsample_unlabeled_data = downsample_unlabeled_data
        if self.downsample_unlabeled_data:
            assert isinstance(downsample_unlabeled_data, float) and 0 < downsample_unlabeled_data < 1
        self.false_positive_rates = false_positive_rates
        self.random_state = random_state


    def run_one_split(self, x, y, x_test, y_test):
        models = {k: None for k in self.training_types}
        model_trainer = {k: None for k in self.training_types}
        metrics = {k: None for k in self.training_types}
        timestamp = int(time())
        
        # split x and y into train and validation sets
        unlabeled_data, labeled_x, _, labeled_y = train_test_split(
            x, y, train_size=self.unlabeled_data_ratio, random_state=self.random_state
        )
        if self.downsample_unlabeled_data:
            # sample N random samples from unlabeled data which is numpy array
            unlabeled_size = unlabeled_data.shape[0]
            indices = np.random.choice(unlabeled_size, int(self.downsample_unlabeled_data*unlabeled_size), replace=False)
            unlabeled_data = unlabeled_data[indices].copy()
        if self.dump_data_splits:
            splitData = f"dataset_splits_{timestamp}.npz"
            np.savez_compressed(
                os.path.join(self.output_dir, splitData),
                unlabeled_data=unlabeled_data,
                labeled_x=labeled_x,
                labeled_y=labeled_y
            )
            logging.warning(f" [!] Saved dataset splits to {splitData}")

        logging.warning(' [!] Pre-training model...')

        # create a pretraining model and task
        models['pretrained'] = self.model_class(**self.model_config).to(self.device)
        model_trainer_config = {
            "device": self.device,
            "model": models['pretrained'],
            "forward_pass": models['pretrained'].pretrain,
            "loss_function": CrossEntropyLoss(),
            "optimizer_class": AdamW,
            "optimizer_config": {"lr": 2.5e-4},
            "optim_scheduler": self.optim_scheduler,
            "optim_step_budget": self.optim_step_budget,
            "verbosity_n_batches": self.verbosity_n_batches,
            "batchSize": self.batch_size,
            "falsePositiveRates": self.false_positive_rates,
        }
        if self.output_dir:
            model_trainer_config["outputFolder"] = os.path.join(self.output_dir, "pretraining")

        self.pretraining_task_config['model_trainer_config'] = model_trainer_config
        self.pretraining_task = self.pretraining_task_class(
            **self.pretraining_task_config
        )
        self.pretraining_task.pretrain(
            unlabeled_data,
            epochs=self.pretrain_pochs,
            dump_model_every_epoch=self.dump_model_every_epoch,
            remask_epochs=self.remask_epochs
        )

        # downstream task for pretrained model
        model_trainer_config['loss_function'] = BCEWithLogitsLoss()
        model_trainer_config['forward_pass'] = None

        for model in self.training_types:
            logging.warning(f' [!] Training {model} model on downstream task...')
            if model != 'pretrained': # if not pretrained -- create a new model
                models[model] = self.model_class(**self.model_config).to(self.device)
            model_trainer_config['model'] = models[model]
            
            if self.output_dir:
                model_trainer_config["outputFolder"] = os.path.join(self.output_dir, f"downstream_task_{model}")
            
            model_trainer[model] = ModelTrainer(**model_trainer_config)
            # TODO: don't like how step is calculated here
            # use size of x and self.unlabeledDataSize to calculate steps
            if model == 'full_data':
                model_trainer_config['optim_step_budget'] = self.optim_step_budget//2
                model_trainer[model].fit(x, y, self.downstream_epochs, reporting_timestamp=timestamp)
            else:
                model_trainer_config['optim_step_budget'] = self.optim_step_budget//10
                model_trainer[model].fit(labeled_x, labeled_y, self.downstream_epochs, reporting_timestamp=timestamp)
        
        for model in models:
            logging.warning(f' [*] Evaluating {model} model on test set...')
            metrics[model] = model_trainer[model].evaluate(x_test, y_test, metrics="json")
            logging.warning(f"\t[!] Test set AUC: {metrics[model]['auc']:.4f}")
            for reportingFPR in self.false_positive_rates:
                f1 = metrics[model]["fpr_"+str(reportingFPR)]["f1"]
                tpr = metrics[model]["fpr_"+str(reportingFPR)]["tpr"]
                logging.warning(f'\t[!] Test set scores at FPR: {reportingFPR:>6} --> TPR: {tpr:.4f} | F1: {f1:.4f}')
        del models, model_trainer # cleanup to not accidentaly reuse 
        clear_cuda_cache()
        return metrics

    def run_splits(self, x, y, x_test, y_test, nSplits=5, rest=None):
        metrics = {k: {"fpr_"+str(fpr): {"f1": [], "tpr": []} for fpr in self.false_positive_rates} for k in self.training_types}
        for trainingType in self.training_types:
            metrics[trainingType]['auc'] = []
        # collect metrics for number of iterations
        for i in range(nSplits):
            self.random_state += i # to get different splits
            logging.warning(f' [!] Running pre-training split {i+1}/{nSplits}')
            splitMetrics = self.run_one_split(x, y, x_test, y_test)
            for trainingType in self.training_types:
                metrics[trainingType]['auc'].append(splitMetrics[trainingType]['auc'])
                for fpr in self.false_positive_rates:
                    metrics[trainingType]["fpr_"+str(fpr)]["f1"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["f1"])
                    metrics[trainingType]["fpr_"+str(fpr)]["tpr"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["tpr"])
            if rest:
                sleep(rest)

        # compute mean and std for each metric
        for trainingType in self.training_types:
            for fpr in self.false_positive_rates:
                metrics[trainingType]["fpr_"+str(fpr)]["f1_mean"] = np.nanmean(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["f1_std"] = np.nanstd(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_mean"] = np.nanmean(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_std"] = np.nanstd(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
        return metrics

