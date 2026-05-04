import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusionLayer(nn.Module):
    """
    A residual MLP layer that approximates a non-linear fusion function.
    Consists of a linear base and a non-linear expansion path.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_linear = nn.Linear(in_dim, out_dim)
        # Non-linear path to capture complex interactions between local and global features
        self.nonlinear_path = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, out_dim),
        )
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # Residual combination of linear and non-linear projections
        base = self.base_linear(x)
        spline = self.nonlinear_path(x)
        return self.layer_norm(base + spline)


class AdaptiveFeatureMixture(nn.Module):
    """
    Per-atom dynamic gating: fuses local (GINE) and global (Mamba) embeddings
    using an adaptive layer to generate importance scores (alpha, beta).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.adaptive_fusion = AdaptiveFusionLayer(d_model * 2, 2)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([local_feat, global_feat], dim=-1)
        gate_raw = self.adaptive_fusion(x)
        gate = F.softmax(gate_raw, dim=-1)
        alpha = gate[:, :1]
        beta = gate[:, 1:]
        return alpha * local_feat + beta * global_feat
