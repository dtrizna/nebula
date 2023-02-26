import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from typing import Optional


class TransformerEncoderChunks(nn.Module):
    def __init__(self,
                    vocab_size: int, # size of vocabulary
                    maxlen: int, # maximum length of input sequence
                    chunk_size: int = 64, # what lengths input sequence should be chunked to
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    norm_first: bool = True, # whether to normalize before or after FFNN layers
                    dropout: float = 0.3,
                    mean_over_sequence=False):
        super().__init__()        
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'TransformerEncoderChunks'
        
        self.encoder = nn.Embedding(vocab_size, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(
            dModel, nHeads, dHidden, dropout, 
            batch_first=True, norm_first=norm_first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        
        self.chunk_size = chunk_size
        self.nr_of_chunks = maxlen/self.chunk_size
        # add 1 if nr_of_chunks is not scalar --> account for the padding
        if self.nr_of_chunks != int(self.nr_of_chunks):
            self.nr_of_chunks = int(self.nr_of_chunks) + 1

        self.meanOverSeq = mean_over_sequence
        if mean_over_sequence:
            input_neurons = dModel
        else:
            input_neurons = int(self.chunk_size * self.nr_of_chunks * dModel)

        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(input_neurons, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i-1], h))

            # add BatchNorm to every layer except last
            if layerNorm and i < len(hiddenNeurons)-1:
                self.ffnnBlock.append(nn.LayerNorm(h))

            self.ffnnBlock.append(nn.ReLU())

            if dropout:
                self.ffnnBlock.append(nn.Dropout(dropout))
            
            self.ffnn.append(nn.Sequential(*self.ffnnBlock))
        self.ffnn = nn.Sequential(*self.ffnn)
        self.fcOutput = nn.Linear(hiddenNeurons[-1], numClasses)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for block in self.ffnn:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.bias.data.zero_()
                    layer.weight.data.uniform_(-initrange, initrange)

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        chunks = []
        for chunk in torch.split(src, split_size_or_sections=self.chunk_size, dim=1):
            if chunk.shape[1] < self.chunk_size:
                pad_mask = (0, self.chunk_size-chunk.shape[1])
                chunk = F.pad(chunk, pad=pad_mask)
    
            chunk = self.encoder(chunk) * math.sqrt(self.d_model)
            chunk = self.pos_encoder(chunk)        
            chunk = self.transformer_encoder(chunk, src_mask)
            # at this stage each chunk is: (batch_size, chunk_size, d_model)
            chunks.append(chunk)
        # after cat it'd be: (batch_size, chunk_size * nr_of_chunks * d_model, d_model)
        # where nr_of_chunks = int(maxLen/self.chunk_size) + 1
        x = torch.cat(chunks, dim=1)
        if self.meanOverSeq:
            x = torch.mean(x, dim=1)
        else:
            x = x.view(x.size(0), -1)
        x = self.ffnn(x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.core(x)
        out = self.fcOutput(x)
        return out


class TransformerEncoderChunksLM(TransformerEncoderChunks):
    def __init__(self,
                    vocab_size: int, # size of vocabulary
                    maxlen: int, # maximum length of input sequence
                    chunk_size: int = 64, # what lengths input sequence should be chunked to
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    norm_first: bool = True, # whether to normalize before or after FFNN layers
                    dropout: float = 0.3,
                    mean_over_sequence=False,
                    # LM specific
                    pretrainLayers: list = [1024] # pretrain layers
                ):
        super().__init__(
            vocab_size=vocab_size,
            maxlen=maxlen,
            chunk_size=chunk_size,
            dModel=dModel,
            nHeads=nHeads,
            dHidden=dHidden,
            nLayers=nLayers,
            numClasses=numClasses,
            hiddenNeurons=hiddenNeurons,
            layerNorm=layerNorm,
            norm_first=norm_first,
            dropout=dropout,
            mean_over_sequence=mean_over_sequence
        )
        self.__name__ = 'TransformerEncoderChunksLM'
        self.preTrainLayers = []
        for i, h in enumerate(pretrainLayers):
            self.preTrainBlock = []
            if i == 0:
                self.preTrainBlock.append(nn.Linear(hiddenNeurons[-1], h))                
            else:
                self.preTrainBlock.append(nn.Linear(pretrainLayers[i-1], h))
            self.preTrainBlock.append(nn.ReLU())
            if dropout:
                self.preTrainBlock.append(nn.Dropout(dropout))
            self.preTrainLayers.append(nn.Sequential(*self.preTrainBlock))
        if self.preTrainLayers:
            self.preTrainLayers.append(nn.Linear(pretrainLayers[-1], vocab_size))
        else:
            self.preTrainLayers.append(nn.Linear(hiddenNeurons[-1], vocab_size))
        self.preTrainLayers = nn.Sequential(*self.preTrainLayers)
    
    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        x = self.preTrainLayers(x)
        return x


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                    vocab_size: int, # size of vocabulary
                    maxlen: int, # maximum length of input sequence
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    norm_first: bool = True, # whether to normalize before or after FFNN layers
                    dropout: float = 0.3):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'TransformerEncoder'
        self.encoder = nn.Embedding(vocab_size, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(
            dModel, nHeads, dHidden, dropout, 
            batch_first=True, norm_first=norm_first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        
        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(self.d_model * maxlen, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i-1], h))

            # add BatchNorm to every layer except last
            if layerNorm and i < len(hiddenNeurons)-1:
                self.ffnnBlock.append(nn.LayerNorm(h))

            self.ffnnBlock.append(nn.ReLU())

            if dropout:
                self.ffnnBlock.append(nn.Dropout(dropout))
            
            self.ffnn.append(nn.Sequential(*self.ffnnBlock))
        self.ffnn = nn.Sequential(*self.ffnn)
        
        self.fcOutput = nn.Linear(hiddenNeurons[-1], numClasses)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for block in self.ffnn:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.bias.data.zero_()
                    layer.weight.data.uniform_(-initrange, initrange)

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        x = x.view(x.size(0), -1)
        x = self.ffnn(x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.core(x)
        out = self.fcOutput(x)
        return out


class TransformerEncoderModelLM(TransformerEncoderModel):
    def __init__(self,
                    vocab_size: int, # size of vocabulary
                    maxlen: int, # maximum length of input sequence
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    norm_first: bool = False, # whether to normalize decoder's FFNN layers
                    dropout: float = 0.3,
                    pretrainLayers: list = [1024], # pretrain layers
    ):
        super().__init__(
            vocab_size=vocab_size,
            maxlen=maxlen,
            dModel=dModel,
            nHeads=nHeads,
            dHidden=dHidden,
            nLayers=nLayers,
            numClasses=numClasses,
            hiddenNeurons=hiddenNeurons,
            layerNorm=layerNorm,
            norm_first=norm_first,
            dropout=dropout
        )
        self.__name__ = 'TransformerEncoderLM'
        self.preTrainLayers = []
        for i, h in enumerate(pretrainLayers):
            self.preTrainBlock = []
            if i == 0:
                self.preTrainBlock.append(nn.Linear(hiddenNeurons[-1], h))                
            else:
                self.preTrainBlock.append(nn.Linear(pretrainLayers[i-1], h))
            self.preTrainBlock.append(nn.ReLU())
            if dropout:
                self.preTrainBlock.append(nn.Dropout(dropout))
            self.preTrainLayers.append(nn.Sequential(*self.preTrainBlock))
        self.preTrainLayers.append(nn.Linear(pretrainLayers[-1], vocab_size))
        self.preTrainLayers = nn.Sequential(*self.preTrainLayers)

    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        x = self.preTrainLayers(x)
        return x


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    """ From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
