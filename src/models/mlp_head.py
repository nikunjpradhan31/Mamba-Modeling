import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 12,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_channels, out_channels))
        else:
            layers.append(nn.Linear(in_channels, hidden_channels))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            layers.append(nn.Linear(hidden_channels, out_channels))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels)
        Returns:
        output: (batch_size, out_channels)
        """
        return self.mlp(x)
