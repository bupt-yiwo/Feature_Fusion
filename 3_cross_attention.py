import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        query:   (B, Q, D) - e.g. learnable query tokens
        context: (B, N, D) - e.g. image patch embeddings
        """
        query2, _ = self.attn(query, context, context)  # Cross-Attention
        query = query + self.dropout(query2)
        return self.norm(query)


B, Q, N, D = 2, 16, 196, 768  # Batch, num_query_tokens, context_len, dim

query = torch.randn(B, Q, D)      # learnable Q-tokens
context = torch.randn(B, N, D)    # image features from ViT

cross_attn = CrossAttention(dim=D)
output = cross_attn(query, context)  # output: (B, Q, D)
