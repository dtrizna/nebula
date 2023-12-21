import torch

class CBOWDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, window_size):
        self.sequences = sequences
        self.window_size = window_size
        
    def __len__(self):
        return sum(len(seq) - (self.window_size * 2) for seq in self.sequences)
    
    def __getitem__(self, idx):
        # find the sequence and the position within the sequence for the sample
        for seq in self.sequences:
            if idx < len(seq) - (self.window_size * 2):
                sequence = seq
                break
            idx -= len(seq) - (self.window_size * 2)
            
        # extract the context and target words for the sample
        context = sequence[idx:idx + self.window_size] + sequence[idx + self.window_size + 1:idx + (self.window_size * 2) + 1]
        target = sequence[idx + self.window_size]
        return context, target



# Define the CBOW model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_size, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
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

class Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes):
        super(Classifier, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, num_classes)
    
    def forward(self, inputs):
        embeddings = self.embeddings(inputs)
        logits = self.linear(embeddings)
        return logits

if __name__ == '__main__':
    # define the model, the loss function, and the optimizer
    model = CBOW(vocab_size, embedding_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create the dataset
    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16]
    ]
    dataset = CBOWDataset(sequences, window_size=2)

    # create the data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # train the model
    for epoch in range(num_epochs):
        for context, target in train_dataloader:
            output = model(context)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # how to setup pre-trained embeddings from classifier
    classifierModel = Classifier(vocab_size, embedding_size, num_classes)
    # get the pre-trained embeddings
    embeddings = model.embedding.weight
    # set the embeddings in the classifier
    classifierModel.embedding.weight = nn.Parameter(embeddings)