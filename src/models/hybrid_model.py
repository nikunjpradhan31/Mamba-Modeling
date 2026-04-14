import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from typing import Callable, Any

from .gin import GINEncoder
from .mamba_model import MambaBlock
from .mlp_head import MLPHead


class GINMambaHybrid(nn.Module):
    def __init__(
        self,
        node_features: int,
        d_model: int,
        gin_hidden: int = 64,
        gin_layers: int = 3,
        mamba_state: int = 16,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        mamba_layers: int = 1,
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        num_tasks: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()

        # GIN encoder for local graph structure
        self.gin = GINEncoder(
            in_channels=node_features,
            hidden_channels=gin_hidden,
            num_layers=gin_layers,
            out_channels=d_model,
            dropout=dropout,
        )

        # Mamba layers for sequence modeling
        self.mamba_layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=mamba_state,
                    d_conv=mamba_conv,
                    expand=mamba_expand,
                )
                for _ in range(mamba_layers)
            ]
        )

        # MLP head for task predictions
        self.mlp = MLPHead(
            in_channels=d_model,
            hidden_channels=mlp_hidden,
            out_channels=num_tasks,
            num_layers=mlp_layers,
            dropout=dropout,
        )

    def forward(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        """
        data: PyG Batch object
        ordering_func: Callable that takes 'data' and returns a node permutation tensor
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Local graph encoding (GIN)
        h = self.gin(x, edge_index)

        # 2. Reordering
        perm = ordering_func(data)
        h = h[perm]
        batch = batch[perm]

        # 3. Form sequences (to dense batch)
        dense_x, mask = to_dense_batch(h, batch)

        # 4. Mamba sequence modeling
        for mamba_layer in self.mamba_layers:
            dense_x = mamba_layer(dense_x)

        # 5. Masked mean pooling over the sequence
        # mask shape: (batch_size, max_seq_len)
        mask_float = mask.float().unsqueeze(-1)  # (batch_size, max_seq_len, 1)
        pooled = (dense_x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(
            min=1e-9
        )

        # 6. Classification
        logits = self.mlp(pooled)
        return logits
