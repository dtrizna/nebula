import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderModel(nn.Module):

    def __init__(self, 
                    nTokens: int, # size of vocabulary
                    dModel: int, # embedding & transformer dimension
                    nHeads: int, # number of heads in nn.MultiheadAttention
                    dHidden: int, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [32], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    dropout: float = 0.5):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'Transformer'
        self.encoder = nn.Embedding(nTokens, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(dModel, nHeads, dHidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        
        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(self.d_model, h))
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
        # also set the bias to zero for the linear layers in self.ffnn
        for block in self.ffnn:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.bias.data.zero_()
                    layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        x = x.mean(dim=1)
        x = self.ffnn(x)
        out = self.fcOutput(x)
        return out


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
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