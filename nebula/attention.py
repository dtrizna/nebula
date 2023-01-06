import math
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Identity
from nebula.reformer import (
    default, Reformer, Always, AbsolutePositionalEmbedding, 
    AxialPositionalEmbedding, FixedPositionalEmbedding, MatrixMultiply
)


class TransformerEncoderModel(nn.Module):
    def __init__(self, 
                    vocabSize: int, # size of vocabulary
                    maxLen: int = 2048, # maximum length of input sequence
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
        encoder_layers = TransformerEncoderLayer(dModel, nHeads, dHidden, dropout)
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
        x = x.view(x.size(0), -1)
        # alternatives: 
        # x = x.mean(dim=1)
        x = self.ffnn(x)
        out = self.fcOutput(x)
        return out


class MyReformerLM(nn.Module):
    def __init__(
        self, 
        vocabSize, 
        maxLen,
        dim = 256, 
        depth = 6, 
        heads = 8, 
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
        post_attn_dropout = 0., 
        layer_dropout = 0., 
        random_rotations_per_head = False, 
        use_scale_norm = False, 
        use_rezero = False, 
        use_full_attn = False, 
        full_attn_thres = 0, 
        reverse_thres = 0, 
        num_mem_kv = 0, 
        one_value_head = False, 
        emb_dim = None, 
        return_embeddings = False, 
        weight_tie_embedding = False, 
        fixed_position_emb = False, 
        absolute_position_emb = False, 
        axial_position_emb = False, 
        axial_position_shape = None, 
        n_local_attn_heads = 0, 
        pkm_layers = tuple(), 
        pkm_num_keys = 128,
        hiddenNeurons: list = [64], # decoder's classifier FFNN complexity
        classifierDropout: int = 0.5,
        numClasses = 1, # binary classification
    ):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = maxLen

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
        return self.preTrainLayers(x_core)
    
    def forward(self, x):
        x = self.core(x)
        x = x.view(x.size(0), -1)
        x = self.ffnn(x)
        return self.fcOutput(x)


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
