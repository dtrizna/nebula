import math
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEncoderChunks(nn.Module):
    def __init__(self,
                 vocab_size: int,  # size of vocabulary
                 maxlen: int,  # maximum length of input sequence
                 chunk_size: int = 64,  # what lengths input sequence should be chunked to
                 dModel: int = 32,  # embedding & transformer dimension
                 nHeads: int = 8,  # number of heads in nn.MultiheadAttention
                 dHidden: int = 200,  # dimension of the feedforward network model in nn.TransformerEncoder
                 nLayers: int = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 numClasses: int = 1,  # 1 ==> binary classification
                 hiddenNeurons: list = [64],  # decoder's classifier FFNN complexity
                 layerNorm: bool = False,  # whether to normalize decoder's FFNN layers
                 norm_first: bool = True,  # whether to normalize before or after FFNN layers
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
        self.nr_of_chunks = maxlen / self.chunk_size
        # add 1 if nr_of_chunks is not scalar --> account for the padding
        if self.nr_of_chunks != int(self.nr_of_chunks):
            self.nr_of_chunks = int(self.nr_of_chunks) + 1

        self.meanOverSeq = mean_over_sequence
        if mean_over_sequence:
            input_neurons = dModel
        else:
            input_neurons = int(self.chunk_size * self.nr_of_chunks * dModel)

        self.ffnn = []
        for i, h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(input_neurons, h))
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
        for block in self.ffnn:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    layer.bias.data.zero_()
                    layer.weight.data.uniform_(-initrange, initrange)

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
        # after .cat(): (batch_size, nr_of_chunks * chunk_size, d_model)
        # where nr_of_chunks = int(maxlen / self.chunk_size) + 1
        x = torch.mean(x, dim=1) if self.meanOverSeq else x.view(x.size(0), -1)
        x = self.ffnn(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.core(x)
        out = self.fcOutput(x)
        return out


class TransformerEncoderOptionalEmbedding(TransformerEncoderChunks):
    def __init__(self,
                 vocab_size: int,  # size of vocabulary
                 maxlen: int,  # maximum length of input sequence
                 chunk_size: int = 64,  # what lengths input sequence should be chunked to
                 dModel: int = 32,  # embedding & transformer dimension
                 nHeads: int = 8,  # number of heads in nn.MultiheadAttention
                 dHidden: int = 200,  # dimension of the feedforward network model in nn.TransformerEncoder
                 nLayers: int = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 numClasses: int = 1,  # 1 ==> binary classification
                 hiddenNeurons: list = [64],  # decoder's classifier FFNN complexity
                 layerNorm: bool = False,  # whether to normalize decoder's FFNN layers
                 norm_first: bool = True,  # whether to normalize before or after FFNN layers
                 dropout: float = 0.3,
                 mean_over_sequence=False,
                 skip_embedding=False):
        super().__init__(vocab_size, maxlen, chunk_size, dModel, nHeads, dHidden, nLayers, numClasses, hiddenNeurons,
                         layerNorm, norm_first, dropout, mean_over_sequence)
        self.skip_embedding = skip_embedding
        self.max_input_length = maxlen

    def core(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        if self.skip_embedding:  # assumes 'src' is already embedded
            x = src
        else:
            x = self.embed(src)
        x = self.transform(x, src_mask)
        x = torch.mean(x, dim=1) if self.meanOverSeq else x.view(x.size(0), -1)
        x = self.ffnn(x)
        return x

    def embed(self, x: Tensor) -> Tensor:
        encoded_chunk = self.encoder(x) * math.sqrt(self.d_model)
        encoded = self.pos_encoder(encoded_chunk)
        return encoded

    def split(self, src: Tensor) -> Tensor:
        return src

    def transform(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        transformed_chunk = self.transformer_encoder(x, src_mask)
        return transformed_chunk
    
    def embed_sample(self, src):
        ssrc = self.split(src)
        return self.embed(ssrc)


class TransformerEncoderChunksLM(TransformerEncoderChunks):
    def __init__(self,
                 vocab_size: int,  # size of vocabulary
                 maxlen: int,  # maximum length of input sequence
                 chunk_size: int = 64,  # what lengths input sequence should be chunked to
                 dModel: int = 32,  # embedding & transformer dimension
                 nHeads: int = 8,  # number of heads in nn.MultiheadAttention
                 dHidden: int = 200,  # dimension of the feedforward network model in nn.TransformerEncoder
                 nLayers: int = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 numClasses: int = 1,  # 1 ==> binary classification
                 hiddenNeurons: list = [64],  # decoder's classifier FFNN complexity
                 layerNorm: bool = False,  # whether to normalize decoder's FFNN layers
                 norm_first: bool = True,  # whether to normalize before or after FFNN layers
                 dropout: float = 0.3,
                 mean_over_sequence=False,
                 # LM specific
                 pretrain_layers: list = [1024]  # pretrain layers
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
        self.pretrain_layers = []
        for i, h in enumerate(pretrain_layers):
            self.preTrainBlock = []
            if i == 0:
                self.preTrainBlock.append(nn.Linear(hiddenNeurons[-1], h))
            else:
                self.preTrainBlock.append(nn.Linear(pretrain_layers[i - 1], h))
            self.preTrainBlock.append(nn.ReLU())
            if dropout:
                self.preTrainBlock.append(nn.Dropout(dropout))
            self.pretrain_layers.append(nn.Sequential(*self.preTrainBlock))
        if self.pretrain_layers:
            self.pretrain_layers.append(nn.Linear(pretrain_layers[-1], vocab_size))
        else:
            self.pretrain_layers.append(nn.Linear(hiddenNeurons[-1], vocab_size))
        self.pretrain_layers = nn.Sequential(*self.pretrain_layers)

    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        x = self.pretrain_layers(x)
        return x


class TransformerEncoderModel(nn.Module):
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
        for i, h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(self.d_model * maxlen, h))
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
                 vocab_size: int,  # size of vocabulary
                 maxlen: int,  # maximum length of input sequence
                 dModel: int = 32,  # embedding & transformer dimension
                 nHeads: int = 8,  # number of heads in nn.MultiheadAttention
                 dHidden: int = 200,  # dimension of the feedforward network model in nn.TransformerEncoder
                 nLayers: int = 2,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                 numClasses: int = 1,  # 1 ==> binary classification
                 hiddenNeurons: list = [64],  # decoder's classifier FFNN complexity
                 layerNorm: bool = False,  # whether to normalize decoder's FFNN layers
                 norm_first: bool = False,  # whether to normalize decoder's FFNN layers
                 dropout: float = 0.3,
                 pretrain_layers: list = [1024],  # pretrain layers
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
        self.pretrain_layers = []
        for i, h in enumerate(pretrain_layers):
            self.preTrainBlock = []
            if i == 0:
                self.preTrainBlock.append(nn.Linear(hiddenNeurons[-1], h))
            else:
                self.preTrainBlock.append(nn.Linear(pretrain_layers[i - 1], h))
            self.preTrainBlock.append(nn.ReLU())
            if dropout:
                self.preTrainBlock.append(nn.Dropout(dropout))
            self.pretrain_layers.append(nn.Sequential(*self.preTrainBlock))
        self.pretrain_layers.append(nn.Linear(pretrain_layers[-1], vocab_size))
        self.pretrain_layers = nn.Sequential(*self.pretrain_layers)

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
