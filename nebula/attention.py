import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Identity
from nebula.reformer import (
    default, Reformer, Always, AbsolutePositionalEmbedding,
    AxialPositionalEmbedding, FixedPositionalEmbedding, MatrixMultiply
)


class TransformerEncoderLM(nn.Module):
    def __init__(self,
                    vocabSize: int, # size of vocabulary
                    maxLen: int, # maximum length of input sequence
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    pretrainLayers: list = [1024], # pretrain layers
                    dropout: float = 0.3):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'Transformer'
        self.encoder = nn.Embedding(vocabSize, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(dModel, nHeads, dHidden, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel

        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(self.d_model * maxLen, h))
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

        # pretrain layers
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
        self.preTrainLayers.append(nn.Linear(pretrainLayers[-1], vocabSize))
        self.preTrainLayers = nn.Sequential(*self.preTrainLayers)


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
    
    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        x = self.preTrainLayers(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.core(x)
        out = self.fcOutput(x)
        return out


class TransformerEncoderModel(nn.Module):
    def __init__(self,
                    vocabSize: int, # size of vocabulary
                    maxLen: int, # maximum length of input sequence
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    dropout: float = 0.5):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'Transformer'
        self.encoder = nn.Embedding(vocabSize, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(dModel, nHeads, dHidden, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        
        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                self.ffnnBlock.append(nn.Linear(self.d_model * maxLen, h))
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

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        x = x.view(x.size(0), -1)
        # alternative -- take mean over sequence length:
        # x = x.mean(dim=1)
        x = self.ffnn(x)
        out = self.fcOutput(x)
        return out


class TransformerEncoderWithChunking(nn.Module):
    def __init__(self,
                    vocabSize: int, # size of vocabulary
                    maxLen: int, # maximum length of input sequence
                    chunk_size: int, # what lengths input sequence should be chunked to
                    dModel: int = 32, # embedding & transformer dimension
                    nHeads: int = 8, # number of heads in nn.MultiheadAttention
                    dHidden: int = 200, # dimension of the feedforward network model in nn.TransformerEncoder
                    nLayers: int = 2, # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
                    numClasses: int = 1, # 1 ==> binary classification 
                    hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
                    layerNorm: bool = False, # whether to normalize decoder's FFNN layers
                    pretrainLayers: list = [1024], # pretrain layers
                    dropout: float = 0.3):
        super().__init__()
        assert dModel % nHeads == 0, "nheads must divide evenly into d_model"
        self.__name__ = 'Transformer_With_Chunking'
        self.encoder = nn.Embedding(vocabSize, dModel)
        self.pos_encoder = PositionalEncoding(dModel, dropout)
        encoder_layers = TransformerEncoderLayer(dModel, nHeads, dHidden, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nLayers)
        self.d_model = dModel
        self.chunk_size = chunk_size
        self.nr_of_chunks = maxLen/self.chunk_size
        # add 1 if nr_of_chunks is not scalar to account for the padding
        if self.nr_of_chunks != int(self.nr_of_chunks):
            self.nr_of_chunks = int(self.nr_of_chunks) + 1

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

        # pretrain layers
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
        self.preTrainLayers.append(nn.Linear(pretrainLayers[-1], vocabSize))
        self.preTrainLayers = nn.Sequential(*self.preTrainLayers)

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

    def core(self, src: Tensor) -> Tensor:
        chunks = []
        for chunk in torch.split(src, split_size_or_sections=self.chunk_size, dim=1):
        # how to pad mask works -- (left, 0, right -- up to chunk_size from existing)
            if chunk.shape[1] < self.chunk_size:
                pad_mask = (0,self.chunk_size-chunk.shape[1])
                chunk = torch.nn.functional.pad(chunk, pad=pad_mask)
    
            chunk = self.encoder(chunk) * math.sqrt(self.d_model)
            chunk = self.pos_encoder(chunk)        
            chunk = self.transformer_encoder(chunk)
            # at this stage each chunk is: (batch_size, chunk_size, d_model)
            chunks.append(chunk)
        # after cat it'd be: (batch_size, chunk_size * nr_of_chunks * d_model, d_model)
        # where nr_of_chunks = int(maxLen/self.chunk_size) + 1
        x = torch.cat(chunks, dim=1)
        x = x.view(x.size(0), -1)
        x = self.ffnn(x)
        return x
    
    def pretrain(self, x: Tensor) -> Tensor:
        x = self.core(x)
        x = self.preTrainLayers(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.core(x)
        out = self.fcOutput(x)
        return out


class ReformerLM(nn.Module):
    def __init__(
        self,
        vocabSize,
        maxLen = None,
        dim = 64,
        depth = 4,
        heads = 4,
        dim_head = 64,
        bucket_size = 64,
        n_hashes = 4,
        ff_chunks = 100,
        attn_chunks = 1,
        causal = False,
        weight_tie = False,
        lsh_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        ff_activation = None,
        ff_glu = False,
        layer_dropout = 0.,
        random_rotations_per_head = False,
        weight_tie_embedding = False,
        use_scale_norm = False,
        use_rezero = False,
        use_full_attn = False,
        full_attn_thres = 0,
        reverse_thres = 0,
        num_mem_kv = 0,
        one_value_head = False,
        emb_dim = None,
        return_embeddings = False,
        fixed_position_emb = False,
        absolute_position_emb = False,
        axial_position_emb = False,
        axial_position_shape = None,
        n_local_attn_heads = 0,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
        classifierDropout: int = 0.5,
        meanOverSequence: bool = True,
        numClasses = 1, # binary classification
    ):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = maxLen
        self.meanOverSeq = meanOverSequence

        self.token_emb = nn.Embedding(vocabSize, emb_dim)

        self.to_model_dim = Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        self.pos_emb = Always(0)
        self.layer_pos_emb = Always(None)

        if axial_position_emb:
            axial_position_shape = default(axial_position_shape, (math.ceil(maxLen / bucket_size), bucket_size))
            self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_position_shape)
        elif absolute_position_emb:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, maxLen)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim)
        else:
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head)

        self.reformer = Reformer(dim, depth, heads = heads, dim_head = dim_head, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, ff_mult = ff_mult, ff_activation = ff_activation, ff_glu = ff_glu, ff_dropout = ff_dropout, post_attn_dropout = 0., layer_dropout = layer_dropout, random_rotations_per_head = random_rotations_per_head, use_scale_norm = use_scale_norm, use_rezero = use_rezero, use_full_attn = use_full_attn, full_attn_thres = full_attn_thres, reverse_thres = reverse_thres, num_mem_kv = num_mem_kv, one_value_head = one_value_head, n_local_attn_heads = n_local_attn_heads, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys)
        self.norm = nn.LayerNorm(dim)

        if return_embeddings:
            self.out = Identity()
            return

        self.preTrainLayers = nn.Sequential(
            nn.Linear(dim, emb_dim) if emb_dim != dim else Identity(),
            nn.Linear(emb_dim, vocabSize) if not weight_tie_embedding else MatrixMultiply(self.token_emb.weight, transpose=True, normalize=True)
        )

        self.ffnn = []
        for i,h in enumerate(hiddenNeurons):
            self.ffnnBlock = []
            if i == 0:
                if self.meanOverSeq:
                    self.ffnnBlock.append(nn.Linear(dim, h))
                else:
                    self.ffnnBlock.append(nn.Linear(dim * maxLen, h))
            else:
                self.ffnnBlock.append(nn.Linear(hiddenNeurons[i-1], h))

            self.ffnnBlock.append(nn.ReLU())

            if classifierDropout:
                self.ffnnBlock.append(nn.Dropout(classifierDropout))
            
            self.ffnn.append(nn.Sequential(*self.ffnnBlock))
        self.ffnn = nn.Sequential(*self.ffnn)
        
        self.fcOutput = nn.Linear(hiddenNeurons[-1], numClasses)

    def core(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.to_model_dim(x)
        x = self.reformer(x, pos_emb = layer_pos_emb, **kwargs)
        x = self.norm(x)
        return x
    
    def pretrain(self, x):
        x_core = self.core(x)
        return self.preTrainLayers(x_core).mean(dim=1)
    
    def forward(self, x):
        x = self.core(x)
        if self.meanOverSeq:
            x = torch.mean(x, dim=1)
        else:
            x = x.view(x.size(0), -1)
        x = self.ffnn(x)
        return self.fcOutput(x)


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
