import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.register_buffer("pe", positional_encoding(max_len, d_model))  # pas un paramètre

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (batch, seq_len, d_model)
        return x + self.pe[: x.size(1)]