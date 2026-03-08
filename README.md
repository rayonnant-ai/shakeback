# Shakeback

Unstick backpropagation with noise perturbation.

When backprop gets stuck in a local minimum, shakeback gives the model a gentle shake — a small random perturbation — and restarts training. The perturbation is proportional to the current loss (`epsilon = loss / 40000`), so it's just enough to escape without destroying what was learned.

## Install

```bash
pip install shakeback
```

## How it works

```
while not done:
    run backprop until patience exhausted
    if improved:
        save best, continue
    else:
        restore best weights
        perturb with epsilon = loss / 40000
        reset optimizer, restart backprop
```

## Quick start

### 1. Define your problem

Create a Python file that tells shakeback how to load your model, data, and compute loss:

```python
# problem.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from shakeback import Problem

class SineProblem(Problem):
    def load_checkpoint(self, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = MyModel().to(device)
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
        torch.save({"model_state_dict": model.state_dict(), **extra}, path)

problem = SineProblem()
```

### 2. Run it

```bash
shakeback --problem problem.py --checkpoint model.pt
```

Or from Python:

```python
from shakeback import shakeback

result = shakeback(
    problem=SineProblem(),
    checkpoint="model.pt",
    patience=10,
    max_shakes=100,
)
print(f"Best loss: {result.best_loss}")
```

See [`examples/sine/`](examples/sine/) for a complete runnable example.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--problem` | required | Path to problem module |
| `--checkpoint` | required | Path to model checkpoint |
| `--output` | `shakeback_best.pt` | Output checkpoint path |
| `--lr` | `1e-4` | AdamW learning rate |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--warmup-epochs` | `3` | LR warmup epochs per run |
| `--max-epochs` | `200` | Max epochs per backprop run |
| `--patience` | `5` | Epochs without improvement before shaking |
| `--max-shakes` | `50` | Max perturbation attempts |
| `--batch-size` | `32` | Training batch size |
| `--val-batches` | `80` | Batches used for validation |
| `--device` | auto | Torch device |

## The Problem interface

Subclass `shakeback.Problem` and implement four methods:

| Method | Returns | Description |
|--------|---------|-------------|
| `load_checkpoint(path, device)` | `(model, dict)` | Load model onto device, return it with any metadata |
| `make_loader(checkpoint_dict, batch_size, device)` | `DataLoader` | Build a training/eval DataLoader |
| `compute_loss(model, batch, device)` | scalar `Tensor` | Forward pass and loss |
| `save_checkpoint(path, model, checkpoint_dict, extra)` | — | Save checkpoint; `extra` has `val_loss`, `total_epochs`, `total_shakes` |

Your module must expose a `problem` instance at module level.

## License

MIT
