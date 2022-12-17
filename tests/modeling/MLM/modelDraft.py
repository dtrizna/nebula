import torch
from torch import nn

# Define the transformer block
class TransformerBlock(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_heads):
    super().__init__()
    
    # Define the linear layers for the self-attention and feed-forward sublayers
    self.attention = nn.Linear(input_dim, hidden_dim)
    self.feed_forward = nn.Linear(input_dim, hidden_dim)
    
    # Define the multi-headed self-attention layer
    self.multi_head_attention = nn.MultiheadAttention(hidden_dim, num_heads)
  
  def forward(self, x):
    # Pass the input through the self-attention sublayer
    x = self.attention(x)
    
    # Pass the output through the multi-headed self-attention layer
    x, _ = self.multi_head_attention(x, x, x)
    
    # Pass the output through the feed-forward sublayer
    x = self.feed_forward(x)
    
    return x


# Define the CNN-transformer model
class CNNLinearTransformerWithPositionalEmbeddings(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_heads, transformer_dim, kernel_sizes, vocab_size, embedding_dim, max_seq_len):
    super().__init__()

    # 
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)
    
    # Define the 1D convolutional layers
    self.convs = nn.ModuleList([nn.Conv1d(input_dim, hidden_dim, kernel_size) for kernel_size in kernel_sizes])

    # Define the linear layer that takes output of the convolutional layers
    self.linear = nn.Linear(sum(kernel_sizes), transformer_dim)

    # Define the transformer block
    self.transformer = TransformerBlock(transformer_dim, transformer_dim, num_heads)

    # Define the linear layer
    self.linear_out = nn.Linear(hidden_dim, 1)

  def forward(self, x):
      # Pass the input through the embedding layer
      x = self.embedding(x)
      x += self.positional_embedding(torch.arange(x.size(1)))
      
      # Transpose the input for the convolutional layers
      x = x.transpose(1, 2)

      # Pass the input through the convolutional layers
      x = torch.cat([conv(x) for conv in self.convs], dim=1)
      
      # Pass the output through the linear layer
      x = self.linear(x)

      # Pass the output through the transformer block
      x = self.transformer(x)
      
      # Pass the output through the linear layer
      x = self.linear_out(x)
      
      return x