import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Projection Head for Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    Projects 768-dim contextual representations to a 128-dim normalized contrastive space.
    MLP Architecture: Linear(input_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim).
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

