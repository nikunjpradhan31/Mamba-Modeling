import torch
import torch.nn as nn
from torch_geometric.nn import GINConv


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(in_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, out_channels),
                    )
                )
            )
        else:
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(in_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    )
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GINConv(
                        nn.Sequential(
                            nn.Linear(hidden_channels, hidden_channels),
                            nn.ReLU(),
                            nn.Linear(hidden_channels, hidden_channels),
                        )
                    )
                )
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, out_channels),
                    )
                )
            )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = torch.relu(x)
                if self.dropout > 0:
                    x = torch.nn.functional.dropout(
                        x, p=self.dropout, training=self.training
                    )
        return x
