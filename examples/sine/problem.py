"""Shakeback problem: learn a sine wave with a small MLP.

No dependencies beyond torch. Run with:

    cd examples/sine
    python make_checkpoint.py
    shakeback --problem problem.py --checkpoint sine_init.pt --patience 3
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shakeback import Problem


class SineNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class SineProblem(Problem):
    def load_checkpoint(self, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = SineNet(hidden=ckpt.get("hidden", 64)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt

    def make_loader(self, checkpoint_dict, batch_size, device):
        x = torch.linspace(-3.14, 3.14, 2000).unsqueeze(1)
        y = torch.sin(x)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                          shuffle=True)

    def compute_loss(self, model, batch, device):
        x, y = batch
        x, y = x.to(device), y.to(device)
        return nn.functional.mse_loss(model(x), y)

    def save_checkpoint(self, path, model, checkpoint_dict, extra):
        torch.save({
            "model_state_dict": model.state_dict(),
            "hidden": checkpoint_dict.get("hidden", 64),
            **extra,
        }, path)


problem = SineProblem()
