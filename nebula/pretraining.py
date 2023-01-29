from nebula import ModelTrainer

import logging
import numpy as np
from tqdm import tqdm

class MaskedLanguageModel(object):
    def __init__(self,
                    vocab,
                    mask_probability=0.15,
                    random_state=None,
                    masked_target_type="onehot"):
        super(MaskedLanguageModel, self).__init__()
        self.__name__ = "MaskedLanguageModel"
        
        self.mask_probability = mask_probability
        self.vocab = vocab
        self.random_state = random_state
        np.random.seed(self.random_state)

        assert masked_target_type in ["onehot", "count"], "masked_target_type must be either 'onehot' or 'count'"
        self.masked_target_type = masked_target_type

    def maskSequence(self, sequence):
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
        maskedTokenIds = np.zeros(vocabSize, dtype=np.int32)

        # limit sequence till first padding token to avoid masking padding
        if self.vocab["<pad>"] in sequence:
            maskedSequence = sequence[:np.where(sequence == self.vocab["<pad>"])[0][0]].copy()
        else:
            maskedSequence = sequence.copy()

        # find out which tokens to mask and loop over    
        maskIdxs = np.random.uniform(size=maskedSequence.shape) < self.mask_probability
        for idx in np.where(maskIdxs)[0]:
            # prepare array of vocabSize that specifies which tokens were masked
            tokenId = maskedSequence[idx]
            if self.masked_target_type == "count":
                maskedTokenIds[tokenId] += 1
            else:
                maskedTokenIds[tokenId] = 1

            # actually mask the token with distribution (80% mask, 10% random, 10% same)
            sample = np.random.sample()
            if sample < 0.8:
                maskedSequence[idx] = self.vocab["<mask>"]
            elif sample < 0.9:
                maskedSequence[idx] = np.random.randint(0, vocabSize)
            else:
                maskedSequence[idx] = sequence[idx]

        # pad masked sequence to be the same length as original sequence
        origSequenceLength = sequence.squeeze().shape[0]
        padWidth = origSequenceLength - maskedSequence.shape[0]
        maskedSequence = np.pad(maskedSequence, (0, padWidth), 'constant', constant_values=self.vocab["<pad>"])

        return maskedSequence, maskedTokenIds

    def maskSequenceArr(self, sequence):
        seq_masked = []
        target = []
        for seq in tqdm(sequence):
            masked_local, target_local = self.maskSequence(seq)
            seq_masked.append(masked_local)
            target.append(target_local)
        # U_masked shape: (n_sequences, max_len)
        # U_target shape: (n_sequences, vocab_size)
        seq_masked = np.vstack(seq_masked)
        target = np.vstack(target)
        return seq_masked, target

    def pretrain(
        self, 
        unlabeledData, 
        modelTrainerConfig, 
        pretrainEpochs=5, 
        dump_model_every_epoch=False,
        remask_every_epoch=False
    ):
        assert isinstance(pretrainEpochs, int), "pretrainEpochs must be an integer"
        modelTrainer = ModelTrainer(**modelTrainerConfig)
        if remask_every_epoch:
            # to avoid dumping results every time
            outFolder = modelTrainer.outputFolder
            modelTrainer.outputFolder = None
            for i in range(pretrainEpochs):
                if i == pretrainEpochs-1:
                    modelTrainer.outputFolder = outFolder
                logging.warning(f' [*] Masking sequences: iteration {i+1}...')
                x_masked, y_masked = self.maskSequenceArr(unlabeledData)
                modelTrainer.fit(x_masked, y_masked, epochs=1, dump_model_every_epoch=dump_model_every_epoch, overwrite_epoch_idx=i)
        else:
            logging.warning(' [*] Masking sequences...')
            x_masked, y_masked = self.maskSequenceArr(unlabeledData)
            modelTrainer.fit(x_masked, y_masked, epochs=pretrainEpochs, dump_model_every_epoch=dump_model_every_epoch)
        return modelTrainer.model
