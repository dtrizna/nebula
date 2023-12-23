from nebula.misc import set_random_seed
from lightning import Trainer

import logging
import numpy as np
from tqdm import tqdm


class MaskedLanguageModelTrainer(Trainer):
    def __init__(self,
                    vocab,
                    model_trainer_config,
                    mask_probability=0.15,
                    random_state=42,
                    masked_target_type="onehot"):
        super().__init__(**model_trainer_config, n_output_classes=len(vocab))
        self.__name__ = "MaskedLanguageModel"
        
        self.mask_probability = mask_probability
        self.vocab = vocab
        self.random_state = random_state
        set_random_seed(self.random_state)

        assert masked_target_type in ["onehot", "count"], "masked_target_type must be either 'onehot' or 'count'"
        self.masked_target_type = masked_target_type

    def mask_seq(self, sequence):
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
            sample = np.random.sample()
            if sample < 0.8:
                masked_sequence[idx] = self.vocab["<mask>"]
            elif sample < 0.9:
                masked_sequence[idx] = np.random.randint(0, vocabSize)
            else:
                masked_sequence[idx] = sequence[idx]

        # pad masked sequence to be the same length as original sequence
        origSequenceLength = sequence.squeeze().shape[0]
        padWidth = origSequenceLength - masked_sequence.shape[0]
        masked_sequence = np.pad(masked_sequence, (0, padWidth), 'constant', constant_values=self.vocab["<pad>"])

        return masked_sequence, masked_token_idxs

    def mask_seq_arr(self, sequence):
        seq_masked = []
        target = []
        for seq in tqdm(sequence):
            masked_local, target_local = self.mask_seq(seq)
            seq_masked.append(masked_local)
            target.append(target_local)
        # U_masked shape: (n_sequences, max_len)
        # U_target shape: (n_sequences, vocab_size)
        seq_masked = np.vstack(seq_masked)
        target = np.vstack(target)
        return seq_masked, target

    def pretrain(
        self,
        x_unlabeled,
        epochs=5,
        time_budget=None,
        dump_model_every_epoch=False,
        remask_epochs=False
    ):
        assert (epochs is not None) ^ (time_budget is not None), \
            "Either 'epochs' or 'time_budget' should be specified"
        assert isinstance(epochs, int), "pretrainEpochs must be an integer"
        if remask_epochs:
            assert isinstance(remask_epochs, int), "remask_epochs must be an integer"
            
            # remove output folder to avoid dumping results every remasking
            orig_out_folder = self.output_folder
            self.output_folder = None

            for i in range(epochs):
                # set output folder back to original value on last iteration
                if i == epochs-1: 
                    self.output_folder = orig_out_folder
                
                if i % remask_epochs == 0:
                    logging.warning(f' [*] Re-masking sequences...')
                    x_masked, y_masked = self.mask_seq_arr(x_unlabeled)
        
                self.fit(
                    x_masked,
                    y_masked,
                    epochs=1,
                    time_budget=time_budget,
                    dump_model_every_epoch=dump_model_every_epoch,
                    overwrite_epoch_idx=i
                )
        else:
            logging.warning(' [*] Masking sequences...')
            x_masked, y_masked = self.mask_seq_arr(x_unlabeled)
            
            self.fit(
                x_masked,
                y_masked,
                epochs=epochs,
                time_budget=time_budget,
                dump_model_every_epoch=dump_model_every_epoch
            )
        
        return self.model


class AutoRegressiveModelTrainer(ModelTrainer):
    def __init__(
            self,
            vocab,
            block_size,
            model_trainer_config,
            random_state=42,
    ):
        super().__init__(**model_trainer_config, n_output_classes=len(vocab))
        self.__name__ = "AutoRegressiveModel"
        self.block_size = block_size

        self.random_state = random_state
        set_random_seed(self.random_state)

    def pretrain(        
        self,
        x_unlabeled,
        epochs=5,
        time_budget=None,
        random_offsets=False,
        dump_model_every_epoch=False,
    ):
        assert (epochs is not None) ^ (time_budget is not None), \
            "Either 'epochs' or 'time_budget' should be specified"
        
        assert isinstance(epochs, int), "epochs must be an integer"
        
        if random_offsets:
            assert isinstance(random_offsets, int), "random_sampling must be an integer"
            ix = np.random.randint(x_unlabeled.shape[0]-self.block_size, size=(random_offsets,))
            x = np.stack([x_unlabeled[i:i+self.block_size] for i in ix])
            y = np.stack([x_unlabeled[i+1:i+self.block_size+1] for i in ix])
        else: # sequential
            x = np.stack([x_unlabeled[i:i+self.block_size] for i in range(x_unlabeled.shape[0]-self.block_size)])
            y = np.stack([x_unlabeled[i+1:i+self.block_size+1] for i in range(x_unlabeled.shape[0]-self.block_size)])

        self.fit(
                x,
                y,
                epochs=epochs,
                time_budget=time_budget,
                dump_model_every_epoch=dump_model_every_epoch
        )
