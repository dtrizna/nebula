from torchtext.data import FastText, Field
import torch.nn as nn

# define the field to process the tokenized sequences
# This field will be used to create a dataset that returns the context and target words for each sample.
field = Field(sequential=True)

# create the dataset
dataset = FastText.splits(field, window_size=2, train=sequences)

# build the vocabulary
field.build_vocab(dataset)


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(CBOWModel, self).__init__()
        
        # define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        # define the linear layer
        self.linear = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, inputs):
        # inputs is a batch of word indices of shape (batch_size, context_size)
        embeddings = self.embeddings(inputs)
        # embeddings is a batch of word embeddings of shape (batch_size, context_size, embedding_size)
        mean_embedding = torch.mean(embeddings, dim=1)
        # mean_embedding is a batch of mean embeddings of shape (batch_size, embedding_size)
        logits = self.linear(mean_embedding)
        # logits is a batch of logits of shape (batch_size, vocab_size)
        return logits


# train and update target model as in cbow