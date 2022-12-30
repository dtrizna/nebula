
import sys
sys.path.append('../..')
from nebula import ModelInterface

import os
import logging
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from sklearn.model_selection import train_test_split

class MaskedLanguageModel(object):
    def __init__(self,
                    vocab,
                    mask_probability=0.15,
                    random_state=None,
                    token_id_type="onehot"):
        super(MaskedLanguageModel, self).__init__()
        
        self.mask_probability = mask_probability
        self.vocab = vocab
        self.random_state = random_state

        if token_id_type not in ("onehot", "count"):
            raise ValueError("token_id_type must be either 'onehot' or 'count'")
        self.token_id_type = token_id_type

        self.__name__ = "MaskedLanguageModel"
    
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
        if self.random_state is not None:
            np.random.seed(self.random_state)
        maskIdxs = np.random.uniform(size=maskedSequence.shape) < self.mask_probability
        for idx in np.where(maskIdxs)[0]:
            # prepare array of vocabSize that specifies which tokens were masked
            tokenId = maskedSequence[idx]
            if self.token_id_type.lower().startswith("count"):
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
        self.seq_masked = []
        self.target = []
        for seq in tqdm(sequence):
            masked_local, target_local = self.maskSequence(seq)
            self.seq_masked.append(masked_local)
            self.target.append(target_local)
        # U_masked shape: (n_sequences, max_len)
        # U_target shape: (n_sequences, vocab_size)
        self.seq_masked = np.vstack(self.seq_masked)
        self.target = np.vstack(self.target)

    def pretrain(self, unlabeledData, modelTrainerConfig, pretrainEpochs=5):
        logging.warning(' [*] Masking sequences...')
        self.maskSequenceArr(unlabeledData)
        # stored in self.seq_masked, self.target
        modelTrainer = ModelInterface(**modelTrainerConfig)
        modelTrainer.fit(self.seq_masked, self.target, epochs=pretrainEpochs)


class SelfSupervisedPretraining:
    def __init__(self, 
                    vocab,
                    modelClass,
                    modelConfig,
                    pretrainingTaskClass,
                    pretrainingTaskConfig,
                    device,
                    falsePositiveRates=[0.001, 0.003, 0.01, 0.03, 0.1],
                    unlabeledDataSize=0.8,
                    randomState=None,
                    pretraingEpochs=5,
                    downstreamEpochs=5,
                    batchSize=256,
                    verbosityBatches=100):
        self.vocab = vocab
        self.modelClass = modelClass
        self.modelConfig = modelConfig
        self.falsePositiveRates = falsePositiveRates
        self.unlabeledDataSize = unlabeledDataSize
        self.randomState = randomState
        self.batchSize = batchSize
        self.pretrainingEpochs = pretraingEpochs
        self.downstreamEpochs = downstreamEpochs
        self.device = device
        self.verbosityBatches = verbosityBatches
        self.pretrainingTask = pretrainingTaskClass(**pretrainingTaskConfig)

        self.trainingTypes = ['pretrained', 'non_pretrained', 'full_data']

    def run(self, x, y, x_test, y_test, outputFolder=None):
        models = {k: None for k in self.trainingTypes}
        modelInterfaces = {k: None for k in self.trainingTypes}
        metrics = {k: None for k in self.trainingTypes}
        
        # split x and y into train and validation sets
        U, L_x, _, L_y = train_test_split(x, y, train_size=self.unlabeledDataSize, random_state=self.randomState)

        # create a pretraining model and task
        logging.warning(' [!] Pre-training model...')
        models['pretrained'] = self.modelClass(**self.modelConfig).to(self.device)
        modelInterfaceConfig = {
            "device": self.device,
            "model": models['pretrained'],
            "modelForwardPass": models['pretrained'].pretrain,
            "lossFunction": CrossEntropyLoss(),
            "optimizerClass": Adam,
            "optimizerConfig": {"lr": 0.001},
            "verbosityBatches": self.verbosityBatches,
            "batchSize": self.batchSize,
            "falsePositiveRates": self.falsePositiveRates,
        }
        if outputFolder:
            modelInterfaceConfig["outputFolder"] = os.path.join(outputFolder, "preTraining")
        self.pretrainingTask.pretrain(U, modelInterfaceConfig, pretrainEpochs=self.pretrainingEpochs)
        
        # downstream task for pretrained model
        modelInterfaceConfig['lossFunction'] = BCEWithLogitsLoss()
        modelInterfaceConfig['modelForwardPass'] = None
        # print torch models parameters

        for model in self.trainingTypes:
            logging.warning(f' [!] Training {model} model on downstream task...')
            if model != 'pretrained':
                models[model] = self.modelClass(**self.modelConfig).to(self.device)
                modelInterfaceConfig['model'] = models[model]
            
            if outputFolder:
                modelInterfaceConfig["outputFolder"] = os.path.join(outputFolder, "downstreamTask_{}".format(model))
            
            modelInterfaces[model] = ModelInterface(**modelInterfaceConfig)
            if model == 'full_data':
                modelInterfaces[model].fit(x, y, self.downstreamEpochs)
            else:
                modelInterfaces[model].fit(L_x, L_y, self.downstreamEpochs)
        
        for model in models:
            logging.warning(f' [*] Evaluating {model} model on test set...')
            metrics[model] = modelInterfaces[model].evaluate(x_test, y_test, metrics="json")
            reportingFPR = self.falsePositiveRates[modelInterfaces[model].fprReportingIdx]
            logging.warning(f' [!] Test F1 score for {model} model at {reportingFPR} FPR : {metrics[model]["fpr_"+str(reportingFPR)]["f1"]:.4f}')
            logging.warning(f' [!] Test TPR score for {model} model at {reportingFPR} FPR: {metrics[model]["fpr_"+str(reportingFPR)]["tpr"]:.4f}')
        del models, modelInterfaces # cleanup to not accidentaly reuse 
        return metrics

    def runSplits(self, x, y, x_test, y_test, outputFolder=None, nSplits=5):
        metrics = {k: {"fpr_"+str(fpr): {"f1": [], "tpr": []} for fpr in self.falsePositiveRates} for k in self.trainingTypes}
        # collect metrics for number of iterations
        for i in range(nSplits):
            self.randomState += i # to get different splits
            logging.warning(f' [!] Running pre-training split {i+1}/{nSplits}')
            splitMetrics = self.run(x, y, x_test, y_test, outputFolder=outputFolder)
            for trainingType in self.trainingTypes:
                for fpr in self.falsePositiveRates:
                    metrics[trainingType]["fpr_"+str(fpr)]["f1"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["f1"])
                    metrics[trainingType]["fpr_"+str(fpr)]["tpr"].append(splitMetrics[trainingType]["fpr_"+str(fpr)]["tpr"])

        # compute mean and std for each metric
        for trainingType in self.trainingTypes:
            for fpr in self.falsePositiveRates:
                metrics[trainingType]["fpr_"+str(fpr)]["f1_mean"] = np.mean(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["f1_std"] = np.std(metrics[trainingType]["fpr_"+str(fpr)]["f1"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_mean"] = np.mean(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
                metrics[trainingType]["fpr_"+str(fpr)]["tpr_std"] = np.std(metrics[trainingType]["fpr_"+str(fpr)]["tpr"])
        return metrics
