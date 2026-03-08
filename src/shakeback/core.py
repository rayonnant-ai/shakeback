"""Shakeback training loop."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from shakeback.problem import Problem


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _evaluate(problem: Problem, model, loader, device, n_batches):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            if n >= n_batches:
                break
            loss = problem.compute_loss(model, batch, device)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def _train_epoch(problem: Problem, model, loader, optimizer, scaler,
                 device, use_amp):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = problem.compute_loss(model, batch, device)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1)


def _make_optimizer(model, lr, weight_decay, warmup_epochs, max_epochs,
                    use_amp):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    return optimizer, scheduler, scaler


def _shake(model, epsilon, device):
    """Apply a random noise perturbation to all model parameters."""
    seed = random.randint(0, 2**31 - 1)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn(param.shape, generator=rng, device=device)
            param.data.add_(noise, alpha=epsilon)
    return seed


@dataclass
class Result:
    """Returned by :func:`shakeback`."""
    best_loss: float
    total_epochs: int
    total_shakes: int


def shakeback(
    problem: Problem,
    checkpoint: str,
    output: str = "shakeback_best.pt",
    *,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3,
    max_epochs: int = 200,
    patience: int = 5,
    max_shakes: int = 50,
    batch_size: int = 32,
    val_batches: int = 80,
    device: torch.device | str | None = None,
) -> Result:
    """Run the shakeback training loop.

    Args:
        problem: Defines model loading, data, loss, and checkpointing.
        checkpoint: Path to the initial checkpoint.
        output: Where to save the best checkpoint.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        warmup_epochs: Linear warmup epochs per backprop run.
        max_epochs: Max epochs per backprop run before stopping.
        patience: Epochs without improvement before shaking.
        max_shakes: Max perturbation attempts before giving up.
        batch_size: Training batch size.
        val_batches: Number of batches used for validation.
        device: Torch device. Auto-detected if None.

    Returns:
        A :class:`Result` with best_loss, total_epochs, and total_shakes.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cuda":
        _log(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        _log(f"Device: {device}")

    # Load
    _log(f"Loading: {checkpoint}")
    model, ckpt_dict = problem.load_checkpoint(checkpoint, device)
    params = sum(p.numel() for p in model.parameters())
    _log(f"Model: {params:,} params")

    loader = problem.make_loader(ckpt_dict, batch_size, device)
    _log(f"Data: {len(loader)} batches/epoch, batch_size={batch_size}")

    use_amp = device.type == "cuda"

    # Initial eval
    best_loss = _evaluate(problem, model, loader, device, val_batches)
    _log(f"Initial eval: {best_loss:.6f}")

    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    total_shakes = 0
    total_epochs = 0

    while total_shakes <= max_shakes:
        optimizer, scheduler, scaler = _make_optimizer(
            model, lr, weight_decay, warmup_epochs, max_epochs, use_amp)

        stale = 0
        run_label = f"run {total_shakes}" if total_shakes > 0 else "initial run"
        _log(f"Starting backprop ({run_label})")

        for epoch in range(1, max_epochs + 1):
            total_epochs += 1
            train_loss = _train_epoch(
                problem, model, loader, optimizer, scaler, device, use_amp)
            scheduler.step()

            val_loss = _evaluate(problem, model, loader, device, val_batches)
            lr_now = scheduler.get_last_lr()[0]
            _log(f"  epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} "
                 f"best={best_loss:.6f} lr={lr_now:.2e}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                problem.save_checkpoint(output, model, ckpt_dict, {
                    "val_loss": best_loss,
                    "total_epochs": total_epochs,
                    "total_shakes": total_shakes,
                })
                _log(f"  ** new best: {best_loss:.6f} (saved)")
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    _log(f"  no improvement for {patience} epochs, shaking")
                    break

        if stale < patience:
            _log("Completed max epochs without stalling")
            break

        total_shakes += 1
        if total_shakes > max_shakes:
            _log(f"Max shakes ({max_shakes}) reached")
            break

        epsilon = best_loss / 40000.0
        model.load_state_dict(best_state)
        seed = _shake(model, epsilon, device)
        _log(f"Shake #{total_shakes}: epsilon={epsilon:.8f} seed={seed}")

    _log(f"Done: best_loss={best_loss:.6f} "
         f"total_epochs={total_epochs} total_shakes={total_shakes}")

    return Result(
        best_loss=best_loss,
        total_epochs=total_epochs,
        total_shakes=total_shakes,
    )
