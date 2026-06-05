"""Visualization helpers for deep beamforming predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy_2ch(tensor_2ch: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor_2ch, torch.Tensor):
        array = tensor_2ch.detach().cpu().numpy()
    else:
        array = np.asarray(tensor_2ch)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[0] != 2:
        raise ValueError(f"Expected tensor with shape [2,H,W] or [1,2,H,W], got {array.shape}")
    return array


def complex_magnitude_from_2ch(tensor_2ch: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return magnitude from a 2-channel complex tensor."""
    array = _to_numpy_2ch(tensor_2ch)
    return np.sqrt(array[0] ** 2 + array[1] ** 2)


def bmode_from_2ch(tensor_2ch: torch.Tensor | np.ndarray, dynamic_range_db: float = 60) -> np.ndarray:
    """Convert a 2-channel complex tensor to a clipped B-mode image in dB."""
    mag = complex_magnitude_from_2ch(tensor_2ch)
    mag_max = float(np.max(mag))
    bmode = 20.0 * np.log10(mag / (mag_max + np.finfo(np.float32).eps) + np.finfo(np.float32).eps)
    return np.clip(bmode, -float(dynamic_range_db), 0.0)


def _axis_1d(axis: torch.Tensor | np.ndarray | None, expected_len: int, name: str) -> np.ndarray | None:
    if axis is None:
        print(f"Warning: {name} not provided; plotting with pixel indices.")
        return None
    if isinstance(axis, torch.Tensor):
        array = axis.detach().cpu().numpy()
    else:
        array = np.asarray(axis)
    array = np.asarray(array, dtype=np.float64).squeeze()
    if array.ndim != 1 or array.size != expected_len or not np.all(np.isfinite(array)):
        print(f"Warning: invalid {name} shape {array.shape}; plotting with pixel indices.")
        return None
    return array


def _imshow_kwargs(image: np.ndarray, x_axis_mm: np.ndarray | None, z_axis_mm: np.ndarray | None) -> dict:
    if x_axis_mm is None or z_axis_mm is None:
        return {"aspect": "auto"}
    return {
        "extent": [
            float(x_axis_mm[0]),
            float(x_axis_mm[-1]),
            float(z_axis_mm[-1]),
            float(z_axis_mm[0]),
        ],
        "origin": "upper",
        "aspect": "auto",
    }


def _label_axes(ax, has_physical_axes: bool) -> None:
    if has_physical_axes:
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Axial [mm]")
    else:
        ax.set_xlabel("W")
        ax.set_ylabel("H")


def save_prediction_comparison(
    y_pred: torch.Tensor | np.ndarray,
    y_target: torch.Tensor | np.ndarray,
    out_path: str | Path,
    title: Optional[str] = None,
    x_axis_mm: torch.Tensor | np.ndarray | None = None,
    z_axis_mm: torch.Tensor | np.ndarray | None = None,
) -> Path:
    """Save a side-by-side comparison of target, prediction, and error."""
    y_pred_np = _to_numpy_2ch(y_pred)
    y_target_np = _to_numpy_2ch(y_target)

    pred_bmode = bmode_from_2ch(y_pred_np)
    target_bmode = bmode_from_2ch(y_target_np)
    error_bmode = np.abs(pred_bmode - target_bmode)
    x_axis = _axis_1d(x_axis_mm, target_bmode.shape[1], "x_axis_mm")
    z_axis = _axis_1d(z_axis_mm, target_bmode.shape[0], "z_axis_mm")
    imshow_kwargs = _imshow_kwargs(target_bmode, x_axis, z_axis)
    has_physical_axes = x_axis is not None and z_axis is not None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    im0 = axes[0].imshow(target_bmode, cmap="gray", vmin=-60, vmax=0, **imshow_kwargs)
    axes[0].set_title("Target B-mode")
    _label_axes(axes[0], has_physical_axes)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_bmode, cmap="gray", vmin=-60, vmax=0, **imshow_kwargs)
    axes[1].set_title("Predicted B-mode")
    _label_axes(axes[1], has_physical_axes)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vmax = float(np.percentile(error_bmode, 99)) if np.isfinite(error_bmode).any() else 1.0
    vmax = max(vmax, 1e-6)
    im2 = axes[2].imshow(error_bmode, cmap="magma", vmin=0, vmax=vmax, **imshow_kwargs)
    axes[2].set_title("|Target - Pred|")
    _label_axes(axes[2], has_physical_axes)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=11)

    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
