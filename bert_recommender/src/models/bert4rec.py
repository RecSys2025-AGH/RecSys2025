import torch
from torch import nn
from dataclasses import dataclass


# GPT said this is a professional way to do it
@dataclass
class BERT4RecConfig:
    # EmbeddingLayer
    vocab_size: int
    embedding_dim: int
    max_seq_len: int
    embedding_dropout: float

    # Encoder
    num_layers: int
    num_heads: int
    hidden_dim: int
    encoder_dropout: float

    # ProjectionHead
    projection_dim: int


class EmbeddingLayer(nn.Module):
    """Item + positional embeddings with layer normalization and dropout"""

    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.item_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch_size, seq_len]
        seq_len = x.size(1)
        # create position ids [0, 1, ..., seq_len - 1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        embeddings = self.item_embeddings(positions) + self.position_embeddings(x)
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)


class Encoder(nn.Module):
    """Transformer encoder. Wrapper for `torch.TransformerEncoderLayer`"""

    def __init__(self, embedding_dim: int, num_layers: int, num_heads: int, hidden_dim: int,
                 dropout: float = 0.1) -> None:
        """

        Args:
            embedding_dim:
            num_layers:
            num_heads:
            hidden_dim:
            dropout:

        Returns:
            None:
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_key_padding_mask=None):
        # x: [batch_size, seq_len, embedding_dim]
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)


class ProjectionHead(nn.Module):
    """Projection head"""

    def __init__(self, embedding_dim: int, projection_dim: int, vocab_size: int):
        super().__init__()
        # transform + layer normalization
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(projection_dim)

        # map to final logits
        self.decoder = nn.Linear(projection_dim, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        proj = self.projection(x)
        proj = self.activation(proj)
        proj = self.layer_norm(proj)
        return self.decoder(proj) + self.bias


class BERT4Rec(nn.Module):
    """BERT4Rec model from `https://arxiv.org/pdf/1904.06690` paper"""

    def __init__(self, config: BERT4RecConfig):
        super().__init__()
        self.embedding = EmbeddingLayer(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            max_seq_len=config.max_seq_len,
            dropout=config.embedding_dropout,
        )
        self.encoder = Encoder(
            embedding_dim=config.embedding_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            dropout=config.encoder_dropout,
        )
        self.projection = ProjectionHead(
            embedding_dim=config.embedding_dim,
            projection_dim=config.projection_dim,
            vocab_size=config.vocab_size,
        )

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.projection(x)
        return x
