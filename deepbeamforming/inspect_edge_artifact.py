"""Inspect right-edge artifacts in overfit-all DBF predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from datasets import ToyDASIQDataset, dbf_collate_fn
    from models import CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad
    from visualize import bmode_from_2ch, complex_magnitude_from_2ch
else:  # pragma: no cover
    from .datasets import ToyDASIQDataset, dbf_collate_fn
    from .models import CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad
    from .visualize import bmode_from_2ch, complex_magnitude_from_2ch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/dbf_baseline/overfit_all_coord_unet_mse/model_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/dbf_baseline/overfit_all_coord_unet_mse/edge_artifact_inspection",
    )
    parser.add_argument("--model", choices=["coord_unet", "coord_unet_reflect", "coord_local_global_reflect"], default="coord_unet")
    return parser.parse_args()


def build_model(name: str, output_size: tuple[int, int]) -> torch.nn.Module:
    if name == "coord_unet":
        return CoordResizeUNetDBF(output_size=output_size)
    if name == "coord_unet_reflect":
        return CoordResizeUNetDBFReflectionPad(output_size=output_size)
    if name == "coord_local_global_reflect":
        return CoordLocalGlobalUNetDBFReflectionPad(output_size=output_size)
    raise ValueError(f"Unsupported model: {name}")


def complex_from_2ch(y: torch.Tensor) -> torch.Tensor:
    return torch.complex(y[:, 0], y[:, 1])


def metrics_for_crop(y_pred: torch.Tensor, y: torch.Tensor, crop_cols: int) -> dict[str, float]:
    if crop_cols > 0:
        y_pred = y_pred[..., crop_cols:-crop_cols]
        y = y[..., crop_cols:-crop_cols]
    pred_c = complex_from_2ch(y_pred)
    target_c = complex_from_2ch(y)
    pred_mag = torch.abs(pred_c)
    target_mag = torch.abs(target_c)
    return {
        "mse": float(torch.mean((y_pred - y) ** 2).item()),
        "mae_magnitude": float(torch.mean(torch.abs(pred_mag - target_mag)).item()),
    }


def column_profiles(y_pred_np: np.ndarray, y_np: np.ndarray) -> dict[str, np.ndarray]:
    pred_c = y_pred_np[0] + 1j * y_pred_np[1]
    target_c = y_np[0] + 1j * y_np[1]
    pred_mag = np.abs(pred_c)
    target_mag = np.abs(target_c)
    return {
        "mae_magnitude_per_column": np.mean(np.abs(pred_mag - target_mag), axis=0),
        "mse_realimag_per_column": np.mean((y_pred_np - y_np) ** 2, axis=(0, 1)),
        "mean_target_magnitude_per_column": np.mean(target_mag, axis=0),
        "mean_prediction_magnitude_per_column": np.mean(pred_mag, axis=0),
        "mae_real_per_column": np.mean(np.abs(y_pred_np[0] - y_np[0]), axis=0),
        "mae_imag_per_column": np.mean(np.abs(y_pred_np[1] - y_np[1]), axis=0),
    }


def save_profile(x: np.ndarray, y: np.ndarray, out_path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    ax.plot(x, y, linewidth=1.4)
    ax.axvline(x[-1] - 4, color="tab:red", linestyle="--", linewidth=0.9, label="last 5 cols")
    ax.axvline(x[-1] - 9, color="tab:orange", linestyle="--", linewidth=0.9, label="last 10 cols")
    ax.axvline(x[-1] - 19, color="tab:purple", linestyle="--", linewidth=0.9, label="last 20 cols")
    ax.set_title(title)
    ax.set_xlabel("Column / lateral index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _extent_kwargs(x_axis_mm: np.ndarray | None, z_axis_mm: np.ndarray | None) -> dict:
    if x_axis_mm is None or z_axis_mm is None:
        print("Warning: physical axes not available for edge artifact maps; using indices.")
        return {"aspect": "auto"}
    return {
        "extent": [float(x_axis_mm[0]), float(x_axis_mm[-1]), float(z_axis_mm[-1]), float(z_axis_mm[0])],
        "origin": "upper",
        "aspect": "auto",
    }


def _label_image_axes(ax, has_axes: bool) -> None:
    if has_axes:
        ax.set_xlabel("Lateral [mm]")
        ax.set_ylabel("Axial [mm]")
    else:
        ax.set_xlabel("W")
        ax.set_ylabel("H")


def save_map_diagnostics(
    y_pred_np: np.ndarray,
    y_np: np.ndarray,
    out_path: Path,
    title: str,
    x_axis_mm: np.ndarray | None = None,
    z_axis_mm: np.ndarray | None = None,
) -> None:
    pred_mag = complex_magnitude_from_2ch(y_pred_np)
    target_mag = complex_magnitude_from_2ch(y_np)
    mag_error = np.abs(pred_mag - target_mag)
    pred_bmode = bmode_from_2ch(y_pred_np)
    target_bmode = bmode_from_2ch(y_np)
    db_diff = pred_bmode - target_bmode
    real_error = y_pred_np[0] - y_np[0]
    imag_error = y_pred_np[1] - y_np[1]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    imshow_kwargs = _extent_kwargs(x_axis_mm, z_axis_mm)
    has_axes = x_axis_mm is not None and z_axis_mm is not None
    ims = []
    ims.append(axes[0, 0].imshow(target_bmode, cmap="gray", vmin=-60, vmax=0, **imshow_kwargs))
    axes[0, 0].set_title("Target B-mode")
    ims.append(axes[0, 1].imshow(pred_bmode, cmap="gray", vmin=-60, vmax=0, **imshow_kwargs))
    axes[0, 1].set_title("Predicted B-mode")
    vmax_mag = max(float(np.percentile(mag_error, 99)), 1e-6)
    ims.append(axes[0, 2].imshow(mag_error, cmap="magma", vmin=0, vmax=vmax_mag, **imshow_kwargs))
    axes[0, 2].set_title("| |pred| - |target| |")
    vmax_db = max(float(np.percentile(np.abs(db_diff), 99)), 1e-6)
    ims.append(axes[1, 0].imshow(db_diff, cmap="coolwarm", vmin=-vmax_db, vmax=vmax_db, **imshow_kwargs))
    axes[1, 0].set_title("Pred - target B-mode [dB]")
    vmax_real = max(float(np.percentile(np.abs(real_error), 99)), 1e-6)
    ims.append(axes[1, 1].imshow(real_error, cmap="coolwarm", vmin=-vmax_real, vmax=vmax_real, **imshow_kwargs))
    axes[1, 1].set_title("Real error")
    vmax_imag = max(float(np.percentile(np.abs(imag_error), 99)), 1e-6)
    ims.append(axes[1, 2].imshow(imag_error, cmap="coolwarm", vmin=-vmax_imag, vmax=vmax_imag, **imshow_kwargs))
    axes[1, 2].set_title("Imag error")
    for ax, im in zip(axes.ravel(), ims):
        _label_image_axes(ax, has_axes)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summarize_edge_concentration(profile: np.ndarray) -> dict[str, float]:
    full_mean = float(np.mean(profile))
    result = {"column_error_mean": full_mean}
    for n in (5, 10, 20):
        edge_mean = float(np.mean(profile[-n:]))
        result[f"last{n}_column_error_mean"] = edge_mean
        result[f"last{n}_over_full_ratio"] = edge_mean / full_mean if full_mean > 0 else float("nan")
    return result


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "beamformed" / "toy_fieldii"
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyDASIQDataset(data_dir)
    h = int(dataset[0]["y"].shape[1])
    w = int(dataset[0]["y"].shape[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(args.model, output_size=(h, w)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model inspection:")
    if args.model in {"coord_unet_reflect", "coord_local_global_reflect"}:
        print("- Reflection-pad model uses ReflectionPad2d(1) + Conv2d(..., padding=0).")
    else:
        print("- CoordResizeUNetDBF uses Conv2d(..., padding=1), i.e. zero padding in convolution layers.")
    print("- Resize and decoder upsampling use interpolate(..., mode='bilinear', align_corners=False).")
    print(f"- model: {args.model}")
    print(f"- checkpoint: {checkpoint_path}")
    print(f"- checkpoint epoch: {checkpoint.get('epoch', 'unknown')}, best_loss: {checkpoint.get('best_loss', 'unknown')}")

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            loader = DataLoader(Subset(dataset, [idx]), batch_size=1, shuffle=False, collate_fn=dbf_collate_fn)
            batch = next(iter(loader))
            x = batch["x"].float().to(device)
            y = batch["y"].float().to(device)
            coords_key = "coords_local_global" if args.model == "coord_local_global_reflect" else "coords"
            coords = batch[coords_key].float().to(device)
            y_pred = model(x, coords=coords)

            y_np = y.squeeze(0).detach().cpu().numpy()
            y_pred_np = y_pred.squeeze(0).detach().cpu().numpy()
            coords_np = coords.squeeze(0).detach().cpu().numpy()
            x_axis_mm = batch.get("x_axis_mm")
            z_axis_mm = batch.get("z_axis_mm")
            x_axis_mm_np = x_axis_mm.squeeze(0).detach().cpu().numpy() if x_axis_mm is not None else None
            z_axis_mm_np = z_axis_mm.squeeze(0).detach().cpu().numpy() if z_axis_mm is not None else None
            file_name = sample["metadata"]["file_name"]
            stem = Path(file_name).stem
            sample_dir = out_dir / stem
            sample_dir.mkdir(parents=True, exist_ok=True)

            profiles = column_profiles(y_pred_np, y_np)
            columns = np.arange(w)
            np.savez_compressed(sample_dir / "column_profiles.npz", **profiles)
            save_profile(
                columns,
                profiles["mae_magnitude_per_column"],
                sample_dir / "error_magnitude_per_column.png",
                f"MAE magnitude per column: {file_name}",
                "MAE magnitude",
            )
            save_profile(
                columns,
                profiles["mean_target_magnitude_per_column"],
                sample_dir / "mean_target_magnitude_per_column.png",
                f"Mean target |IQ| per column: {file_name}",
                "mean |target|",
            )
            save_profile(
                columns,
                profiles["mean_prediction_magnitude_per_column"],
                sample_dir / "mean_prediction_magnitude_per_column.png",
                f"Mean prediction |IQ| per column: {file_name}",
                "mean |prediction|",
            )
            save_profile(
                columns,
                profiles["mse_realimag_per_column"],
                sample_dir / "mse_realimag_per_column.png",
                f"MSE real/imag per column: {file_name}",
                "MSE real/imag",
            )
            save_profile(
                columns,
                profiles["mae_real_per_column"],
                sample_dir / "mae_real_per_column.png",
                f"MAE real per column: {file_name}",
                "MAE real",
            )
            save_profile(
                columns,
                profiles["mae_imag_per_column"],
                sample_dir / "mae_imag_per_column.png",
                f"MAE imag per column: {file_name}",
                "MAE imag",
            )
            save_map_diagnostics(
                y_pred_np,
                y_np,
                sample_dir / "edge_artifact_maps.png",
                f"Edge artifact diagnostics: {file_name}",
                x_axis_mm=x_axis_mm_np,
                z_axis_mm=z_axis_mm_np,
            )

            crop_metrics = {}
            for crop in (0, 5, 10, 20):
                m = metrics_for_crop(y_pred, y, crop)
                suffix = "full" if crop == 0 else f"crop{crop}"
                crop_metrics[f"mse_{suffix}"] = m["mse"]
                crop_metrics[f"mae_magnitude_{suffix}"] = m["mae_magnitude"]

            edge = summarize_edge_concentration(profiles["mae_magnitude_per_column"])
            row = {
                "file_name": file_name,
                "coord_source": sample["metadata"].get("coord_source", "unknown"),
                "x_coord_min": float(coords_np[1].min()),
                "x_coord_max": float(coords_np[1].max()),
                "z_coord_min": float(coords_np[0].min()),
                "z_coord_max": float(coords_np[0].max()),
                "artifact_dir": str(sample_dir),
                **crop_metrics,
                **edge,
            }
            rows.append(row)

    csv_path = out_dir / "edge_artifact_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = out_dir / "edge_artifact_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print("\nCrop metrics and coordinate ranges:")
    header = (
        f"{'sample':38s} {'mse_full':>10s} {'mse_c5':>10s} {'mse_c10':>10s} {'mse_c20':>10s} "
        f"{'mae_full':>10s} {'mae_c5':>10s} {'mae_c10':>10s} {'mae_c20':>10s} "
        f"{'last5/full':>10s} {'x_min':>7s} {'x_max':>7s} {'z_min':>7s} {'z_max':>7s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['file_name']:38s} "
            f"{row['mse_full']:10.6f} {row['mse_crop5']:10.6f} {row['mse_crop10']:10.6f} {row['mse_crop20']:10.6f} "
            f"{row['mae_magnitude_full']:10.6f} {row['mae_magnitude_crop5']:10.6f} "
            f"{row['mae_magnitude_crop10']:10.6f} {row['mae_magnitude_crop20']:10.6f} "
            f"{row['last5_over_full_ratio']:10.3f} "
            f"{row['x_coord_min']:7.2f} {row['x_coord_max']:7.2f} {row['z_coord_min']:7.2f} {row['z_coord_max']:7.2f}"
        )
    print(f"\nsaved metrics: {json_path}")
    print(f"saved csv: {csv_path}")
    print(f"saved plots under: {out_dir}")


if __name__ == "__main__":
    main()
