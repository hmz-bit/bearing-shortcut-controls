"""WDCNN of the Vieira et al. repository, re-implemented from the text of
bearing-data-leakage/src/models/components/backbone/wdcnn.py (not imported, not executed).

Layers (file lines in comments refer to that file):
  5 x [Conv1d(bias=False) -> BatchNorm1d -> MaxPool1d(2, 2) -> ReLU]           (class ConvBnPool1D)
     block1: Conv1d(1, 16, k=64, stride=16, padding=24)                        (lines 40-42)
     block2: Conv1d(16, 32, k=3, s=1, p=1); block3: (32, 64); block4: (64, 64); block5: (64, 64, padding=0)
  Flatten -> LazyLinear(100) -> BatchNorm1d(100) -> ReLU                       (f3 = relu(bn(fc(f2))))
  (dropout off: configs/experiment/paderborn/eval_envelope.yaml apply_dropout: False) -> Linear(100, n_class=2)
Loss: torch.nn.BCEWithLogitsLoss (configs/modelmodule/model/class_loss/multilabel.yaml); sigmoid is only implicit in the loss,
the network outputs logits.  Optimizer: torch.optim.Adam, lr 1e-3 (eval_envelope.yaml: lr 1e-3, batch_size 32).
"""
import torch
from torch import nn
import torch.nn.functional as F


class ConvBnPool1D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(cout)
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        return F.relu(self.pool(self.bn(self.conv(x))))


class WDCNN(nn.Module):
    def __init__(self, n_in=21000, n_class=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBnPool1D(1, 16, kernel_size=64, stride=16, padding=24),
            ConvBnPool1D(16, 32), ConvBnPool1D(32, 64), ConvBnPool1D(64, 64), ConvBnPool1D(64, 64, padding=0))
        with torch.no_grad():
            nflat = self.conv_layers(torch.zeros(2, 1, n_in)).flatten(1).shape[1]
        self.nflat = nflat
        self.fc = nn.Linear(nflat, 100)     # LazyLinear(100) in the original
        self.bn = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, n_class)

    def forward(self, x):
        f = torch.flatten(self.conv_layers(x), 1)
        return self.out(F.relu(self.bn(self.fc(f))))


if __name__ == "__main__":
    for n in (21000, 12000):
        m = WDCNN(n)
        print(n, "flatten", m.nflat, "params", sum(p.numel() for p in m.parameters()))
