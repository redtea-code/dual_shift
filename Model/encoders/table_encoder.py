"""Table encoder for tabular clinical data."""
import torch.nn as nn


class TableEncoder(nn.Module):
    """Encodes tabular features into causal and confounding representations."""

    def __init__(self, input_dim, out_dim, num_classes=3):
        super().__init__()
        self.causal_encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * out_dim),
            nn.ReLU(),
            nn.Linear(2 * out_dim, out_dim)
        )
        self.conf_encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * out_dim),
            nn.ReLU(),
            nn.Linear(2 * out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x):
        """x: (B, D) -> (causal_repr, conf_repr)"""
        return self.causal_encoder(x), self.conf_encoder(x)
