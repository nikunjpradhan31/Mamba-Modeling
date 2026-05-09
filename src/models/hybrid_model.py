import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter
from typing import Callable, Any

from .gin import GINEncoder
from .mamba_model import MambaBlock
from .bidirectional_mamba import BiMambaBlock, create_bidirectional_mamba_layers
from .mlp_head import MLPHead
from .fusion_layer import AdaptiveFeatureMixture, BilinearAttentionFusion, SqueezeExcitationFusion, GLUHighwayFusion, LateFusionLayer


class GINMambaHybrid(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        gin_out_channels: int = 64,
        mamba_d_model: int = 64,
        fusion_dim: int = 64,
        gin_hidden: int = 128,
        gin_layers: int = 4,
        mamba_state: int = 64,
        mamba_conv: int = 4,
        mamba_expand: int = 2,
        mamba_layers: int = 1,
        bidirectional: bool = False,
        model_type: str = "hybrid",
        mlp_hidden: int = 64,
        mlp_layers: int = 2,
        num_tasks: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.model_type = model_type

        if model_type in ("hybrid", "gin"):
            self.gin = GINEncoder(
                in_channels=node_features,
                hidden_channels=gin_hidden,
                num_layers=gin_layers,
                out_channels=gin_out_channels,
                dropout=dropout,
            ) 
        else:
            self.gin = None

        # Projection layer to map combined node and edge features to mamba_d_model
        self.raw_feature_proj = nn.Linear(node_features + edge_features, mamba_d_model)
        
        self.gin_proj = nn.Linear(gin_out_channels, fusion_dim) if model_type in ("hybrid", "gin") else None
        self.mamba_proj = nn.Linear(mamba_d_model, fusion_dim)

        if bidirectional and mamba_layers > 0:
            self.mamba_layers = create_bidirectional_mamba_layers(
                d_model=mamba_d_model,
                d_state=mamba_state,
                d_conv=mamba_conv,
                expand=mamba_expand,
                num_layers=mamba_layers,
            )
        else:
            self.mamba_layers = nn.ModuleList(
                [
                    MambaBlock(
                        d_model=mamba_d_model,
                        d_state=mamba_state,
                        d_conv=mamba_conv,
                        expand=mamba_expand,
                    )
                    for _ in range(mamba_layers)
                ]
            )
        
        self.fusion_layer = AdaptiveFeatureMixture(fusion_dim)
        self.mlp = MLPHead(
            in_channels= fusion_dim,
            hidden_channels=mlp_hidden,
            out_channels=num_tasks,
            num_layers=mlp_layers,
            dropout=dropout,
        )



    def encode_atoms_global(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        """Global stream: Mamba features only."""
        x, batch = data.x, data.batch
        edge_index, edge_attr = data.edge_index, data.edge_attr

        # Aggregate edge features into nodes using PyG scatter
        # We use scatter to sum attributes, then divide by degree for mean
        if edge_attr is not None:
            edge_context = scatter(edge_attr[edge_index[0]], edge_index[1], dim=0, dim_size=x.size(0), reduce='mean')
        else:
            # Fallback to zeros if no edge features are present
            edge_context = torch.zeros((x.size(0), (self.raw_feature_proj.in_features - x.size(-1))), device=x.device)
            
        combined_x = torch.cat([x, edge_context], dim=-1)
        
        if len(self.mamba_layers) == 0:
            return self.raw_feature_proj(combined_x)

        perm_output = ordering_func(data, descending=False)
        if isinstance(perm_output, tuple):
            perm, scores = perm_output
            raw_features = self.raw_feature_proj(combined_x) * scores.unsqueeze(-1)
        else:
            perm = perm_output
            raw_features = self.raw_feature_proj(combined_x)

        inv_perm = torch.argsort(perm)
        batch_perm = batch[perm]
        dense_x, mask = to_dense_batch(raw_features[perm], batch_perm)

        for mamba_layer in self.mamba_layers:
            dense_x = mamba_layer(dense_x)

        mask_expanded = mask.unsqueeze(-1).expand_as(dense_x)
        h_mamba_ordered = dense_x[mask_expanded].view(-1, dense_x.size(-1))
        return h_mamba_ordered[inv_perm]

    def encode_atoms(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)

        if self.model_type == "mamba":
            return self.mamba_proj(self.encode_atoms_global(data, ordering_func))

        if self.model_type in ("hybrid", "gin"):
            h = self.gin(x, edge_index, edge_attr=edge_attr)
        else:
            h = None

        if len(self.mamba_layers) == 0 and self.model_type == "gin": 
            return self.gin_proj(h) if self.gin_proj else h

        h_mamba = self.encode_atoms_global(data, ordering_func)

        if self.model_type == "hybrid":
            # Ensure both streams are projected to fusion_dim
            h_proj = self.gin_proj(h) if self.gin_proj else h
            h_mamba_proj = self.mamba_proj(h_mamba)
            return self.fusion_layer(h_proj, h_mamba_proj)
        else:
            # Gin-only model (with Mamba layers > 0) or other
            return self.mamba_proj(h_mamba)

    def forward(self, data: Any, ordering_func: Callable) -> torch.Tensor:
        """Late fusion: Separate streams until final logits."""

        h_fused = self.encode_atoms(data, ordering_func)
        pooled = global_mean_pool(h_fused, data.batch)
        logits = self.mlp(pooled)
        return logits
