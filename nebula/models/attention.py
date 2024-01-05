import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderModel(nn.Module):
    def __init__(
            self,
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
            dropout: float = 0.3,
            pooling: str = "mean",
            skip_embedding: bool = False,
            pretrain_layers: Optional[List] = None
    ):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        assert pooling in ["mean", "cls", None]
        self.pooling_type = pooling
        self.__name__ = 'TransformerEncoderModel'
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.skip_embedding = skip_embedding
        self.encoder = nn.Embedding(vocab_size, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model=dModel,
            nhead=nHeads,
            dim_feedforward=dHidden,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        self.layerNorm = layerNorm
        self.dropout = dropout
        self.ffnn_layers_in = hiddenNeurons

        if pooling == None:
            input_neurons = int(self.maxlen * dModel)
        if pooling == "mean":
            input_neurons = self.d_model
        self.ffnn_layers = self._build_ffnn_layers(self.ffnn_layers_in, input_neurons)
        self.ffnn = nn.Sequential(*self.ffnn_layers)
        self.ffnn_out_size = hiddenNeurons[-1]
        if numClasses == 2: # binary classification
            self.classifier_head = nn.Linear(self.ffnn_out_size, 1)
        else:
            self.classifier_head = nn.Linear(self.ffnn_out_size, numClasses)

        self.apply(self._init_weights)

        if pretrain_layers is not None:
            if len(pretrain_layers) > 0:
                self.pretrain_layers = self._build_ffnn_layers(pretrain_layers, self.ffnn_out_size)
                self.pretrain_layers.append(nn.Linear(pretrain_layers[-1], self.vocab_size, bias=False))
            else:
                self.pretrain_layers = [nn.Linear(self.ffnn_out_size, self.vocab_size, bias=False)]
            # NOTE: bias in last layer removed the same as in https://github.com/karpathy/nanoGPT/blob/master/model.py#L133
            self.pretrain_layers = nn.Sequential(*self.pretrain_layers)
        else:
            self.pretrain_layers = None

    def _init_weights(self, module):
        # NOTE: from: https://github.com/karpathy/nanoGPT/blob/master/model.py#L162
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_ffnn_layers(self, layers: list, start_neurons: int) -> nn.Sequential:
        ffnn = []
        for i, h in enumerate(layers):
            ffnnBlock = []
            if i == 0:
                ffnnBlock.append(nn.Linear(start_neurons, h))
            else:
                ffnnBlock.append(nn.Linear(layers[i - 1], h))

            # add LayerNorm to every layer except last
            if self.layerNorm and i < len(layers) - 1:
                ffnnBlock.append(nn.LayerNorm(h))

            ffnnBlock.append(nn.ReLU())
            ffnnBlock.append(nn.Dropout(self.dropout))

            ffnn.append(nn.Sequential(*ffnnBlock))
        return ffnn

    def embed(self, x: Tensor) -> Tensor:
        encoded_x = self.encoder(x) * math.sqrt(self.d_model)
        encoded = self.pos_encoder(encoded_x)
        return encoded

    def pooling(self, x: Tensor) -> Tensor:
        if self.pooling_type == None:
            x = x.view(x.size(0), -1)
        if self.pooling_type == "mean":
            x = torch.mean(x, dim=1)
        return x

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.embed(src) if not self.skip_embedding else src
        x = self.transformer_encoder(x, src_mask)
        x = self.pooling(x)
        x = self.ffnn(x)
        return x

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        x = self.core(x, src_mask)
        out = self.classifier_head(x)
        return out

    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        x = self.pretrain_layers(x)
        return x



class TransformerEncoderChunks(TransformerEncoderModel):
    """
    Slices global attention to multiple separate chunks, which are processed independently till classifier head.
    Main model used in: https://arxiv.org/abs/2310.10664
    'Chunk' term borrowed from longformer:
    https://github.com/allenai/longformer/blob/master/longformer/sliding_chunks.py
    """

    def __init__(
            self,
            # size of local attention
            chunk_size: int = 64,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__name__ = 'TransformerEncoderChunks'

        self.chunk_size = chunk_size
        self.nr_of_chunks = self.maxlen / self.chunk_size
        if self.nr_of_chunks != int(self.nr_of_chunks):
            self.nr_of_chunks = int(self.nr_of_chunks) + 1
        
        if self.pooling_type == None: # rebuild FFNN for case when pooling == None
            input_neurons = int(self.chunk_size * self.nr_of_chunks * self.d_model)
            self.ffnn_layers = self._build_ffnn_layers(self.ffnn_layers_in, input_neurons)
            self.ffnn = nn.Sequential(*self.ffnn_layers)


    def split(self, src: Tensor) -> List[Tensor]:
        chunks = []
        for chunk in torch.split(src, split_size_or_sections=self.chunk_size, dim=1):
            if chunk.shape[1] < self.chunk_size:
                pad_mask = (0, self.chunk_size - chunk.shape[1])
                chunk = F.pad(chunk, pad=pad_mask)
            chunks.append(chunk)
        return chunks

    def embed(self, chunks: List[Tensor]) -> List[Tensor]:
        encoded_chunks = []
        for chunk in chunks:
            encoded_chunk = self.encoder(chunk) * math.sqrt(self.d_model)
            encoded_chunk = self.pos_encoder(encoded_chunk)
            encoded_chunks.append(encoded_chunk)
        return encoded_chunks

    def transform(self, chunks: List[Tensor], src_mask: Optional[Tensor] = None) -> List[Tensor]:
        transformed_chunks = []
        for chunk in chunks:
            transformed_chunk = self.transformer_encoder(chunk, src_mask)
            transformed_chunks.append(transformed_chunk)
        return transformed_chunks

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        chunks = self.split(src)
        chunks = self.embed(chunks)  # [(batch_size, chunk_size, d_model), ..]
        chunks = self.transform(chunks, src_mask)  # [(batch_size, chunk_size, d_model), ..]
        x = torch.cat(chunks, dim=1)
        # NOTE: after .cat() shape is: (batch_size, nr_of_chunks * chunk_size, d_model)
        # where nr_of_chunks = int(maxlen / self.chunk_size) + 1
        x = self.pooling(x)
        x = self.ffnn(x)
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
