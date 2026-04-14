import torch
import torch.nn as nn
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        Returns:
        output: (batch_size, seq_len, d_model)
        """
        return self.mamba(x)
