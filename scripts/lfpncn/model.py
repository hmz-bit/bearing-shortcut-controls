"""LF-PNCN: three separate identical conv stems (OR, IR, Ball) -> concat (96) -> three sigmoid heads."""
import torch
from torch import nn


def _stem():
    return nn.Sequential(
        nn.Conv1d(15, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
        nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
        nn.Conv1d(32, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(),
    )


class LFPNCN(nn.Module):
    """Input (B, 3, 15, 64) ordered OR, IR, Ball. Output logits (B, 3); use BCEWithLogitsLoss.
    probs = torch.sigmoid(logits)."""

    def __init__(self):
        super().__init__()
        self.stems = nn.ModuleList([_stem() for _ in range(3)])
        self.heads = nn.ModuleList([nn.Linear(96, 1) for _ in range(3)])

    def forward(self, x):
        z = torch.cat([s(x[:, i]) for i, s in enumerate(self.stems)], dim=1)
        return torch.cat([h(z) for h in self.heads], dim=1)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.sigmoid(self(x))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
