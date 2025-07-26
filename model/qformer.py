import torch
import torch.nn as nn

class QFormerLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1, add_cross_attention=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.add_cross = add_cross_attention
        if self.add_cross:
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x, image_feats=None):
        # Self-Attention
        x_res = x
        x = self.norm1(x)
        sa_out, _ = self.self_attn(x, x, x)
        x = x_res + sa_out

        # Cross-Attention
        if self.add_cross and image_feats is not None:
            x_res = x
            x = self.norm2(x)
            ca_out, _ = self.cross_attn(x, image_feats, image_feats)
            x = x_res + ca_out

        # Feedforward
        x = x + self.ffn(x)
        return x


class QFormer(nn.Module):
    def __init__(
        self,
        num_query_tokens=32,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        num_layers=12,
        cross_attention_frequency=2,
        dropout=0.1,
    ):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            add_cross = (i % cross_attention_frequency == 0)
            self.layers.append(QFormerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                add_cross_attention=add_cross,
            ))
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, image_feats):
        """
        image_feats: [B, N, D] from frozen vision encoder
        returns: [B, Q, D] fused query embeddings
        """
        B = image_feats.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, Q, D]
        for layer in self.layers:
            queries = layer(queries, image_feats)
        return self.norm(queries)
