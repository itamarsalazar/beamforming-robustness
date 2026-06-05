"""Quick forward/backward check for baseline deep beamforming models."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from datasets import ToyDASIQDataset, dbf_collate_fn
    from models import ResizeConvDBF, CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad
else:  # pragma: no cover
    from .datasets import ToyDASIQDataset, dbf_collate_fn
    from .models import ResizeConvDBF, CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad


def check_one_model(
    model_name: str,
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    coords: torch.Tensor | None = None,
) -> None:
    model.train()
    y_pred = model(x, coords=coords)
    loss = F.l1_loss(y_pred, y)

    print(f"model: {model_name}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"y_pred shape: {tuple(y_pred.shape)}")
    if coords is not None:
        print(f"coords shape: {tuple(coords.shape)}")
    print(f"initial L1 loss: {loss.item():.6f}")

    assert y_pred.shape == y.shape

    loss.backward()
    grad_ok = all(param.grad is not None for param in model.parameters() if param.requires_grad)
    print(f"backward ok: {grad_ok}")
    assert grad_ok


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "beamformed" / "toy_fieldii"
    dataset = ToyDASIQDataset(data_dir)

    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dbf_collate_fn)
    batch = next(iter(loader))

    x = batch["x"].float()
    y = batch["y"].float()
    coords = batch.get("coords")
    if coords is not None:
        coords = coords.float()
        print(f"coord sources: {[m.get('coord_source', 'unknown') for m in batch['metadata']]}")
    coords_local_global = batch.get("coords_local_global")
    if coords_local_global is not None:
        coords_local_global = coords_local_global.float()
        print(f"coords_local_global shape: {tuple(coords_local_global.shape)}")
    output_size = (int(y.shape[2]), int(y.shape[3]))

    models = {
        "resizeconv": ResizeConvDBF(output_size=output_size),
        "coord_unet": CoordResizeUNetDBF(output_size=output_size),
        "coord_unet_reflect": CoordResizeUNetDBFReflectionPad(output_size=output_size),
        "coord_local_global_reflect": CoordLocalGlobalUNetDBFReflectionPad(output_size=output_size),
    }

    for model_name, model in models.items():
        model_coords = coords_local_global if model_name == "coord_local_global_reflect" else coords
        check_one_model(model_name, model, x, y, coords=model_coords)
        print()


if __name__ == "__main__":
    main()
