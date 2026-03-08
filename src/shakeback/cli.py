"""Command-line interface for shakeback."""

from __future__ import annotations

import argparse
import importlib.util
import sys

from shakeback.core import shakeback
from shakeback.problem import Problem


def load_problem(path: str) -> Problem:
    """Import a problem module and return its ``problem`` instance."""
    spec = importlib.util.spec_from_file_location("_problem", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "problem"):
        obj = mod.problem
        if isinstance(obj, Problem):
            return obj
        raise SystemExit(
            f"ERROR: 'problem' in {path} is {type(obj).__name__}, "
            f"expected a shakeback.Problem instance")

    # Check for legacy function-based interface
    for fn in ("load_checkpoint", "make_loader", "compute_loss",
               "save_checkpoint"):
        if not hasattr(mod, fn):
            raise SystemExit(
                f"ERROR: {path} must define a `problem` instance "
                f"(subclass of shakeback.Problem) or all four functions: "
                f"load_checkpoint, make_loader, compute_loss, save_checkpoint")

    # Wrap function-based module in a Problem
    class _Wrapped(Problem):
        def load_checkpoint(self, p, device):
            return mod.load_checkpoint(p, device)

        def make_loader(self, ckpt_dict, batch_size, device):
            return mod.make_loader(ckpt_dict, batch_size, device)

        def compute_loss(self, model, batch, device):
            return mod.compute_loss(model, batch, device)

        def save_checkpoint(self, p, model, ckpt_dict, extra):
            return mod.save_checkpoint(p, model, ckpt_dict, extra)

    return _Wrapped()


def main():
    parser = argparse.ArgumentParser(
        prog="shakeback",
        description="Unstick backpropagation with noise perturbation.")
    parser.add_argument("--problem", type=str, required=True,
                        help="Path to problem module (.py)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="shakeback_best.pt",
                        help="Output checkpoint path (default: shakeback_best.pt)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="LR warmup epochs per run (default: 3)")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Max epochs per backprop run (default: 200)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Epochs without improvement before shaking (default: 5)")
    parser.add_argument("--max-shakes", type=int, default=50,
                        help="Max perturbation attempts (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--val-batches", type=int, default=80,
                        help="Validation batches (default: 80)")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device (default: auto)")
    args = parser.parse_args()

    problem = load_problem(args.problem)

    shakeback(
        problem=problem,
        checkpoint=args.checkpoint,
        output=args.output,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        max_shakes=args.max_shakes,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        device=args.device,
    )
