import numpy as np
from collections import Counter

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

    # generate array that specifies which elements were masked
    whereMasked = np.zeros(origSequenceLength, dtype=np.int8)
    whereMasked[np.where(maskIdxs)] = 1

    return maskedSequence, whereMasked, maskedTokenIds


def maskSequenceArr(sequence, vocab, mask_probability=0.15, random_state=None, token_id_type="onehot"):
    """
    Should be faster than maskSequence, but does not work as expected -- see TODO
    """
    vocabSize=len(vocab)
    origSequenceLength = sequence.shape[0]

    # limit sequence till first padding token
    if vocab["<pad>"] in sequence:
        maskedSequence = sequence[:np.where(sequence == vocab["<pad>"])[0][0]].copy()
    else:
        maskedSequence = sequence.copy()
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # get indices of values to mask according to probability setting
    maskIdxs = np.random.uniform(size=maskedSequence.shape) < mask_probability
    # filter those elemets only
    maskedSequenceFiltered = maskedSequence[maskIdxs]
    # get probabilities for mask heuristic: whether to mask, randomize or keep the same
    subMaskProbs = np.random.uniform(size=maskedSequenceFiltered.shape) 
    # use nested np.where to modify maskedSequence
    maskedSequence[maskIdxs] = np.where(subMaskProbs < 0.8, vocab["<mask>"], # if subMaskProbs < 0.8, then vocab["<mask>"]
            # TODO: inputs the same value for all 0.8-0.9, should be random each time
            np.where(subMaskProbs < 0.9, np.random.randint(0, vocabSize), # if subMaskProbs < 0.9, then random int
            maskedSequenceFiltered)) # else preserve the same token
    
    # pad masked sequence to be the same length as original sequence
    maskedSequence = np.pad(maskedSequence, (0, origSequenceLength - maskedSequence.shape[0]), 'constant', constant_values=vocab["<pad>"])
    
    # preprare array that specifies which elements were masked
    whereMasked = np.zeros(shape=sequence.shape, dtype=np.int8)
    whereMasked[np.where(maskIdxs)] = 1
    
    # prepare array of vocabSize that specifies which tokens were masked
    removedTokens = sequence[np.where(whereMasked)]
    maskedTokenIds = np.zeros(shape=(vocabSize), dtype=np.int8)
    if token_id_type.lower().startswith("count"):
        counter = Counter(removedTokens)
        maskedTokenIds[list(counter.keys())] = list(counter.values())
    else:
        maskedTokenIds[removedTokens] = 1


    return maskedSequence, whereMasked, maskedTokenIds

