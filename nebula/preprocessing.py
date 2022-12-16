from collections import Counter

def buildVocab(tokenListSequence, vocab_size=10000, getCounter=False):
    counter = Counter()
    for tokenList in tokenListSequence:
        counter.update(tokenList)
    
    specialTokens = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
    vocab = {x[0]:i+len(specialTokens) for i,x in enumerate(counter.most_common(vocab_size-len(specialTokens)))}
    vocab.update(specialTokens)
    if getCounter:
        return vocab, counter
    else:
        return vocab

def labelEncoder(tokenizedList, vocab):
    tokenizedListEncoded = []
    for tokenized in tokenizedList:
        tokenizedEncoded = [vocab[x] if x in vocab else vocab["<unk>"] for x in tokenized]
        tokenizedListEncoded.append(tokenizedEncoded)
    return tokenizedListEncoded

def padSequence(sequence, max_seq_len, pad_token):
    if len(sequence) > max_seq_len:
        sequence = sequence[:max_seq_len]
    else:
        sequence += [pad_token] * (max_seq_len - len(sequence))
    return sequence

def padSequenceList(sequenceList, max_seq_len, pad_token):
    sequenceListPadded = []
    for sequence in sequenceList:
        sequenceListPadded.append(padSequence(sequence, max_seq_len, pad_token))
    return sequenceListPadded
