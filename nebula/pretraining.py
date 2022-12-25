import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../..')
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nebula.evaluation import get_tpr_at_fpr

import logging
from tqdm import tqdm
from pandas import DataFrame
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# Now I need to convert all these function in a single class that use common parameters through self
class SSLPretraining:
    def __init__(self, x, y, x_test, y_test, vocab, modelClass, modelConfig, fprs=[0.0001, 0.001, 0.01, 0.1], test_size=0.2, random_state=42, batchSize=32, pretraingEpochs=10, downstreamEpochs=10, device='cpu', verbosityBatches=50, mask_probability=0.15):
        # TBD -- change so class uses self
        self.x = x
        self.y = y
        self.x_test = x_test
        self.y_test = y_test
        self.vocab = vocab
        self.modelClass = modelClass
        self.modelConfig = modelConfig
        self.fprs = fprs
        self.test_size = test_size
        self.random_state = random_state
        self.batchSize = batchSize
        self.pretrainingEpochs = pretraingEpochs
        self.downstreamEpochs = downstreamEpochs
        self.device = device
        self.verbosityBatches = verbosityBatches
        self.mask_probability = mask_probability


    def pretrain(self, U_masked, U_target, model, batchSize, pretrinEpochs, device, verbosityBatches=50):
        # make a loader from U_masked and U_target
        preTrainLoader = DataLoader(
            # create dataset from U_masked and U_target numpy arrays
            TensorDataset(torch.from_numpy(U_masked), torch.from_numpy(U_target)),
            batch_size=batchSize,
            shuffle=True
        )

        lossFunction = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # pre-train model
        for epoch in range(1, pretrinEpochs+1):
            for batch_idx, (data, target) in enumerate(preTrainLoader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                
                # forward pass
                pred_masked_vocab = model.pretrain(data)
                loss = lossFunction(pred_masked_vocab, target.float())
                
                loss.backward()
                optimizer.step()

                if batch_idx % verbosityBatches == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(preTrainLoader.dataset),
                        100. * batch_idx / len(preTrainLoader), loss.item()))


    def downstream(self, L_x, L_y, model, batchSize, downstreamEpochs, device, verbosityBatches=50):
        # make a loader from L_x and L_y
        downstreamLoader = DataLoader(
            # create dataset from L_x and L_y numpy arrays
            TensorDataset(torch.from_numpy(L_x), torch.from_numpy(L_y)),
            batch_size=batchSize,
            shuffle=True
        )

        # train model
        logging.warning('Training model...')
        lossFunction = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

        for epoch in range(1, downstreamEpochs+1):
            for batch_idx, (data, target) in enumerate(downstreamLoader):
                data, target = data.to(device), target.to(device).reshape(-1, 1)
                optimizer.zero_grad()
                
                # forward pass
                pred = model(data)
                loss = lossFunction(pred, target.float())
                loss.backward()
                optimizer.step()
                
                predProbs = torch.sigmoid(pred).cpu().detach().numpy()
                f1 = f1_score(target.cpu().detach().numpy(), predProbs > 0.5, average='macro')
                if batch_idx % verbosityBatches == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(downstreamLoader.dataset),
                        100. * batch_idx / len(downstreamLoader), loss.item(), f1))



    def evaluate(self, model, x_test, y_test, fprs, device, batchSize):
        testLoader = DataLoader(
                TensorDataset(torch.from_numpy(x_test).long(), torch.from_numpy(y_test).float()),
                batch_size=batchSize,
                shuffle=True
            )
        model.eval()

        lossFunction = nn.BCEWithLogitsLoss()
        metrics = defaultdict(lambda: defaultdict(list))
        for data, target in tqdm(testLoader):
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                logits = model(data)

            loss = lossFunction(logits, target.float().reshape(-1,1))

            y_pred_probs = torch.sigmoid(logits).clone().detach().cpu().numpy()
            target = target.clone().detach().cpu().numpy().reshape(-1,1)
            for fpr in fprs:
                tpr_at_fpr, threshold_at_fpr = get_tpr_at_fpr(target, y_pred_probs, fpr)
                # f1 = f1_score(y[test_index], predicted_probs >= threshold)
                f1_at_fpr = f1_score(target, y_pred_probs >= threshold_at_fpr)
                metrics[fpr]["tpr"].append(tpr_at_fpr)
                metrics[fpr]["f1"].append(f1_at_fpr)
                metrics[fpr]["loss"].append(loss.item())
        # take the mean of the metrics
        for fpr in metrics:
            metrics[fpr] = DataFrame(metrics[fpr])
            # add std for each metric - tpr, f1, loss
            metrics[fpr]["tpr_std"] = metrics[fpr]["tpr"].std()
            metrics[fpr]["f1_std"] = metrics[fpr]["f1"].std()
            metrics[fpr]["loss_std"] = metrics[fpr]["loss"].std()
            # take the mean of the metrics
            metrics[fpr] = metrics[fpr].mean(axis=0)
            metrics[fpr] = metrics[fpr].to_dict()
        return metrics

    def run(self, x, y, x_test, y_test, vocab, modelClass, modelConfig, fprs=[0.0001, 0.001, 0.01, 0.1], test_size=0.2, random_state=42, batchSize=256, pretraingEpochs=1, downstreamEpochs=1, device='cpu', verbosityBatches=50, mask_probability=0.15):
        # split x and y into train and validation sets
        U, L_x, _, L_y = train_test_split(x, y, test_size=test_size, random_state=random_state)

        # for each sequence in U, mask it
        logging.warning('Masking sequences...')
        U_masked, U_target = maskSequenceArr(U, vocab, mask_probability=mask_probability, random_state=None)

        # pre-train model
        logging.warning('Pre-training model...')
        model_Pretrained = modelClass(**modelConfig)
        model_Pretrained.to(device)
        self.pretrain(U_masked, U_target, model_Pretrained, batchSize, pretraingEpochs, device, verbosityBatches)

        # downstream task for pretrained model
        logging.warning('Training pre-trained model on downstream task...')
        self.downstream(L_x, L_y, model_Pretrained, batchSize, downstreamEpochs, device, verbosityBatches)
        
        # downstream task for new model
        logging.warning('Training new model on downstream task...')
        model_NonPretrained = modelClass(**modelConfig)
        model_NonPretrained.to(device)
        self.downstream(L_x, L_y, model_NonPretrained, batchSize, downstreamEpochs, device, verbosityBatches)

        # downstream task for new model on full dataset suitable for benchmarking
        logging.warning('Training new model on downstream task on full dataset...')
        model_Full = modelClass(**modelConfig)
        model_Full.to(device)
        self.downstream(x, y, model_Full, batchSize, downstreamEpochs, device, verbosityBatches)

        logging.warning('Evaluating all models on test set...')
        # get fpr and f1 on test set on pretrained model
        metrics_Pretrained = self.evaluate(model_Pretrained, x_test, y_test, fprs, device, batchSize)
        # get performance metrics on fresh model
        metrics_nonPretrained = self.evaluate(model_NonPretrained, x_test, y_test, fprs, device, batchSize)
        # get performance metrics on full model
        metrics_full = self.evaluate(model_Full, x_test, y_test, fprs, device, batchSize)
        
        metrics = dict(zip(
                    ['pretrained_U_Lx', 'non_pretrained_Lx', 'full_data_X'], 
                    [metrics_Pretrained, metrics_nonPretrained, metrics_full]
                ))
        return metrics



def maskSequence(sequence, vocab, mask_probability=0.15, random_state=None, token_id_type="onehot"):
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
    vocabSize=len(vocab)
    maskedTokenIds = np.zeros(vocabSize, dtype=np.int32)

    # limit sequence till first padding token to avoid masking padding
    if vocab["<pad>"] in sequence:
        maskedSequence = sequence[:np.where(sequence == vocab["<pad>"])[0][0]].copy()
    else:
        maskedSequence = sequence.copy()

    # find out which tokens to mask and loop over
    if random_state is not None:
        np.random.seed(random_state)
    maskIdxs = np.random.uniform(size=maskedSequence.shape) < mask_probability
    for idx in np.where(maskIdxs)[0]:
    
        # prepare array of vocabSize that specifies which tokens were masked
        tokenId = maskedSequence[idx]
        if token_id_type.lower().startswith("count"):
            maskedTokenIds[tokenId] += 1
        else:
            maskedTokenIds[tokenId] = 1

        # actually mask the token
        sample = np.random.sample()
        if sample < 0.8:
            maskedSequence[idx] = vocab["<mask>"]
        elif sample < 0.9:
            maskedSequence[idx] = np.random.randint(0, vocabSize)
        else:
            maskedSequence[idx] = sequence[idx]

    # pad masked sequence to be the same length as original sequence
    origSequenceLength = sequence.squeeze().shape[0]
    padWidth = origSequenceLength - maskedSequence.shape[0]
    maskedSequence = np.pad(maskedSequence, (0, padWidth), 'constant', constant_values=vocab["<pad>"])

    # REMOVED: doesn't make sense -- maskedSequence already represents this information
    # generate array that specifies which elements were masked
    # whereMasked = np.zeros(origSequenceLength, dtype=np.int8)
    # whereMasked[np.where(maskIdxs)] = 1

    return maskedSequence, maskedTokenIds


def maskSequenceArr(sequence, vocab, mask_probability=0.15, random_state=None, token_id_type="onehot"):
    seq_masked, target = [], []
    for seq in tqdm(sequence):
        masked_local, target_local = maskSequence(seq, vocab, mask_probability, random_state, token_id_type)
        seq_masked.append(masked_local)
        target.append(target_local)
    # U_masked shape: (n_sequences, max_len)
    # U_target shape: (n_sequences, vocab_size)
    return np.vstack(seq_masked), np.vstack(target)