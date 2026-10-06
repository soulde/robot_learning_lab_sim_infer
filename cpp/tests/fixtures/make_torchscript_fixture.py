#!/usr/bin/env python3
"""Create deterministic TorchScript modules used only by native CTest."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


class AffineSlice(torch.nn.Module):
    def __init__(self, out_dim: int, multiplier: float, bias: float):
        super().__init__()
        self.out_dim = out_dim
        self.multiplier = multiplier
        self.register_buffer("bias", torch.full((out_dim,), bias, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, : self.out_dim] * self.multiplier + self.bias


class TupleModel(torch.nn.Module):
    def forward(self, value: torch.Tensor):
        return value[:, :2], value[:, :2]


class WrongDtype(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :2].to(torch.int64)


class NonFinite(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, :2] * float("nan")


def save(module: torch.nn.Module, sample: torch.Tensor, path: Path) -> None:
    torch.jit.trace(module.eval(), sample).save(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(AffineSlice(2, 1.0, 1.0), torch.zeros(1, 3), args.output_dir / "velocity.pt")
    save(AffineSlice(4, 2.0, 1.0), torch.zeros(1, 5), args.output_dir / "tracking.pt")
    save(TupleModel(), torch.zeros(1, 3), args.output_dir / "tuple.pt")
    save(AffineSlice(2, 1.0, 1.0), torch.zeros(1, 3), args.output_dir / "wrong_shape.pt")
    save(WrongDtype(), torch.zeros(1, 3), args.output_dir / "wrong_dtype.pt")
    save(NonFinite(), torch.zeros(1, 3), args.output_dir / "non_finite.pt")
    (args.output_dir / "corrupt.pt").write_text("not a torchscript module", encoding="utf-8")


if __name__ == "__main__":
    main()
