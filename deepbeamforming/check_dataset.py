"""Quick verification for the toy DAS-IQ PyTorch dataset."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from datasets import ToyDASIQDataset, dbf_collate_fn
else:  # pragma: no cover
    from .datasets import ToyDASIQDataset, dbf_collate_fn


def complex_rms_from_2ch(tensor: torch.Tensor) -> float:
    complex_tensor = torch.complex(tensor[0], tensor[1])
    return float(torch.sqrt(torch.mean(torch.abs(complex_tensor) ** 2)).item())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "beamformed" / "toy_fieldii"
    dataset = ToyDASIQDataset(data_dir)

    print(f"Dataset size: {len(dataset)}")

    for idx in range(len(dataset)):
        item = dataset[idx]
        x = item["x"]
        y = item["y"]
        meta = item["metadata"]

        x_rms = complex_rms_from_2ch(x)
        y_rms = complex_rms_from_2ch(y)

        x_has_nan = not torch.isfinite(x).all().item()
        y_has_nan = not torch.isfinite(y).all().item()

        print(
            f"{meta['file_name']} | x {tuple(x.shape)} | y {tuple(y.shape)} | "
            f"x_rms={x_rms:.6f} | y_rms={y_rms:.6f} | NaN/Inf x={x_has_nan} y={y_has_nan}"
        )

        assert not x_has_nan
        assert not y_has_nan
        assert abs(x_rms - 1.0) < 1e-5
        assert abs(y_rms - 1.0) < 1e-5

    loader1 = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dbf_collate_fn)
    batch1 = next(iter(loader1))
    print(f"DataLoader batch_size=1 -> x {tuple(batch1['x'].shape)} | y {tuple(batch1['y'].shape)}")

    loader2 = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dbf_collate_fn)
    batch2 = next(iter(loader2))
    print(f"DataLoader batch_size=2 -> x {tuple(batch2['x'].shape)} | y {tuple(batch2['y'].shape)}")

    assert batch1["x"].ndim == 4
    assert batch1["y"].ndim == 4
    assert batch2["x"].ndim == 4
    assert batch2["y"].ndim == 4


if __name__ == "__main__":
    main()
