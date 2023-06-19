import torch
from torch import nn

import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CharEmbedding(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(
            self.in_chans,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        # Input x shape: [B, C, L]
        x = self.proj(x)  # [B, E, L']
        x = x.transpose(1, 2)  # [B, L', E]
        return x


class CharTransformer(nn.Module):
    def __init__(self,
                patch_size = 16,
                in_chans = 1,
                embed_dim = 128,
                nHeads = 8,
                dHidden = 256,
                transformer_layers = 2,
                numClasses: int = 1,  # 1 ==> binary classification
                hiddenNeurons: list = [64],  # decoder's classifier FFNN complexity
                layerNorm: bool = False,
                dropout: float = 0.3
                ):
        super().__init__()
        self.patch_embed = CharEmbedding(patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # learnable parameter
        self.pos_embed = nn.Parameter(torch.zeros(1, 1000 + 1, embed_dim))  # learnable parameter
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nHeads,
                dim_feedforward=dHidden,
                dropout=0.3,
                activation="relu",
                batch_first=True,
                norm_first=True,
                
            ),
            num_layers=transformer_layers
        )
        self.ffnn = []
        for i, h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(embed_dim, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i - 1], h))

            # add BatchNorm to every layer except last
            if layerNorm and i < len(hiddenNeurons) - 1:
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
        self.pos_embed.data.uniform_(-initrange, initrange)
        self.cls_token.data.uniform_(-initrange, initrange)
        for block in self.ffnn:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.bias.data.zero_()
                    layer.weight.data.uniform_(-initrange, initrange)

    def embed(self, x):
        x = x.type(torch.cuda.FloatTensor)
        B, _ = x.shape
        x = x.unsqueeze(1)  # [B, C, L] where C is 1
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+L', E]

        x = x + self.pos_embed[:, :(x.size(1)), :]  # add positional encoding
        return x

    def core(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        # take cls token
        x = x[:, 0, :]  # [B, E]
        x = self.ffnn(x)
        return x

    def forward(self, x):
        x = self.core(x)
        x = self.fcOutput(x)
        return x


class TransformerEncoderModelCls(nn.Module):
    def __init__(self,
                 vocab_size: int,  # size of vocabulary
                 maxlen: int,  # maximum length of input sequence
                 dModel: int = 32,  # embedding & transformer dimension
                 nHeads: int = 8,  # number of heads in nn.MultiheadAttention
                 dHidden: int = 200,  # dimension of the feedforward network model in nn.TransformerEncoder
                 nLayers: int = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 numClasses: int = 1,  # 1 ==> binary classification
                 hiddenNeurons: list = [64],  # decoder's classifier FFNN complexity
                 layerNorm: bool = False,  # whether to normalize decoder's FFNN layers
                 norm_first: bool = True,  # whether to normalize before or after FFNN layers
                 dropout: float = 0.3):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'TransformerEncoderCls'
        self.encoder = nn.Embedding(vocab_size, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dModel))  # learnable parameter
        encoder_layers = TransformerEncoderLayer(
            dModel, nHeads, dHidden, dropout,
            batch_first=True, norm_first=norm_first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel

        self.ffnn = []
        for i, h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(self.d_model, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i - 1], h))

            # add BatchNorm to every layer except last
            if layerNorm and i < len(hiddenNeurons) - 1:
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
        self.cls_token.data.uniform_(-initrange, initrange)
        for block in self.ffnn:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.bias.data.zero_()
                    layer.weight.data.uniform_(-initrange, initrange)

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        
        # add cls token
        cls_tokens = self.cls_token.expand(src.size(0), -1, -1)  # [B, L, E]
        src = torch.cat((cls_tokens, src), dim=1)  # [B, 1+L', E]
        
        src = self.pos_encoder(src)
        
        x = self.transformer_encoder(src, src_mask)
        # grab only attentions of cls token
        x = x[:, 0, :]  # [B, E]
        
        x = self.ffnn(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.core(x)
        out = self.fcOutput(x)
        return out


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
