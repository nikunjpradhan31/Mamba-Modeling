import torch
import torch.nn as nn
from src.models.gin import GINEncoder


class LearnedOrdering(nn.Module):
    """
    Learned node ordering using an auxiliary GNN.
    Returns both the permutation indices and the continuous scores.
    The continuous scores are needed to multiply with node features
    so gradients can flow back into this module, since argsort is non-differentiable.
    """

    def __init__(
        self,
        node_features: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Use a lightweight GIN to score each node
        self.gnn = GINEncoder(
            in_channels=node_features,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=1,  # 1D score per node
            dropout=dropout,
        )

    def forward(self, data, descending=False, **kwargs):
        device = data.batch.device
        num_nodes = data.num_nodes

        # Calculate scores
        # Output is [num_nodes, 1]
        raw_scores = self.gnn(data.x, data.edge_index)
        raw_scores = raw_scores.squeeze(-1)  # [num_nodes]

        # Bound scores to [0, 1]
        scores = torch.sigmoid(raw_scores)

        # Add small noise for tie-breaking
        noise = torch.rand(num_nodes, device=device, dtype=torch.float32) * 1e-5
        noisy_scores = scores + noise

        if descending:
            noisy_scores = -noisy_scores

        # To ensure we sort within each graph in the batch,
        # we add the batch index to the normalized scores.
        # Since noisy_scores could slightly exceed 1 or go below 0 due to noise,
        # we strictly normalize them to (0, 1) before adding the batch index.
        norm_scores = noisy_scores - noisy_scores.min()
        max_score = norm_scores.max()
        if max_score > 0:
            norm_scores = norm_scores / (max_score + 1e-5)

        batch_scores = data.batch.to(torch.float32) + norm_scores

        perm = torch.argsort(batch_scores)

        # Return both permutation and continuous scores for soft gating
        return perm, scores
