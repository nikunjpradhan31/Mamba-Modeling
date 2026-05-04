import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaptiveFeatureMixture(nn.Module):
    """
    Per-atom dynamic gating to fuse local (GIN) and global (Mamba) embeddings.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.fusion = AdaptiveFusionLayer(d_model)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        return self.fusion(local_feat, global_feat)


class AdaptiveFusionLayer(nn.Module):
    """
    Computes dynamic gating weights for local and global features.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([local_feat, global_feat], dim=-1)
        alpha, beta = self.gate(gate_input).chunk(2, dim=-1)
        return alpha * local_feat + beta * global_feat


class BilinearAttentionFusion(nn.Module):
    """
    Computes scalar compatibility scores between local and global features.
    Injects interaction signals into a gating network.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.bilinear = nn.Bilinear(d_model, d_model, 1)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        interaction = self.bilinear(local_feat, global_feat).squeeze(-1)
        gate_input = torch.cat([local_feat, global_feat, interaction.unsqueeze(-1)], dim=-1)
        alpha, beta = self.gate(gate_input).chunk(2, dim=-1)
        return alpha * local_feat + beta * global_feat


class SqueezeExcitationFusion(nn.Module):
    """
    Modulates local features using global context via excitation vectors.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.excite = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        global_pooled = global_feat.mean(dim=0, keepdim=True)
        excitation = self.excite(global_pooled)
        return local_feat * excitation + global_feat


class GLUHighwayFusion(nn.Module):
    """
    Gated Linear Unit (GLU) style fusion with residual connections.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([local_feat, global_feat], dim=-1)
        gate = F.sigmoid(self.gate(gate_input))
        fused = self.proj(gate_input)
        return gate * fused + (1 - gate) * local_feat


class LateFusionLayer(nn.Module):
    """
    Late fusion layer for combining logits from two streams (local and global).
    Learns per-task weights to combine logits.
    """
    def __init__(self, num_tasks: int = 12):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(num_tasks, 2))

    def forward(self, local_logits: torch.Tensor, global_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_logits: (batch_size, num_tasks) from local stream
            global_logits: (batch_size, num_tasks) from global stream
        Returns:
            Combined logits: (batch_size, num_tasks)
        """
        # Softmax over the weight dimension (local vs global)
        w = F.softmax(self.weights, dim=-1)  # (num_tasks, 2)
        # Expand weights to (1, num_tasks, 2) for broadcasting
        w = w.unsqueeze(0)
        # Stack logits: (batch_size, num_tasks, 2)
        logits_stack = torch.stack([local_logits, global_logits], dim=-1)
        # Weighted sum: (batch_size, num_tasks)
        return (w * logits_stack).sum(dim=-1)