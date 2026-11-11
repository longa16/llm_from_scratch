import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import create_embedding_layer, embed_tokens
from positional_encoding import positional_encoding


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_projection = nn.Linear(d_model, 3 * d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_projection(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.output_dropout(self.output_projection(attended))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention_norm(x + self.attention(x))
        return self.feed_forward_norm(x + self.feed_forward(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        d_ff: int | None = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or d_model <= 0 or num_layers <= 0 or max_seq_len <= 0:
            raise ValueError("vocab_size, d_model, num_layers, and max_seq_len must be positive")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by a positive num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1)")

        self.max_seq_len = max_seq_len
        self.token_embedding = create_embedding_layer(vocab_size, d_model)
        positions = torch.from_numpy(positional_encoding(max_seq_len, d_model)).float()
        self.register_buffer("position_encoding", positions.unsqueeze(0), persistent=False)
        self.embedding_dropout = nn.Dropout(dropout)
        feed_forward_size = d_ff if d_ff is not None else 4 * d_model
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, num_heads, feed_forward_size, dropout)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")
        seq_len = token_ids.size(1)
        if seq_len == 0 or seq_len > self.max_seq_len:
            raise ValueError("sequence_length must be between 1 and max_seq_len")

        x = embed_tokens(self.token_embedding, token_ids, self.token_embedding.embedding_dim)
        x = self.embedding_dropout(x + self.position_encoding[:, :seq_len].to(dtype=x.dtype))
        for block in self.blocks:
            x = block(x)
        return self.output_projection(self.final_norm(x))

    @torch.no_grad()
    def generate(
        self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0
    ) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.size(1) == 0:
            raise ValueError("token_ids must be a non-empty 2D tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        generated = token_ids
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.max_seq_len :]
                next_token_logits = self(context)[:, -1, :] / temperature
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)
        return generated