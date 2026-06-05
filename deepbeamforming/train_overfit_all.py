"""Overfit the five toy DAS-IQ samples with the CoordResizeUNetDBF baseline."""

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
    from visualize import save_prediction_comparison
else:  # pragma: no cover
    from .datasets import ToyDASIQDataset, dbf_collate_fn
    from .models import CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad
    from .visualize import save_prediction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--model", choices=["coord_unet", "coord_unet_reflect", "coord_local_global_reflect"], default="coord_unet")
    parser.add_argument("--loss-type", choices=["mse"], default="mse")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm; <=0 disables it")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/dbf_baseline/overfit_all_coord_unet_mse",
        help="Output directory relative to repo root, unless absolute",
    )
    return parser.parse_args()


def build_model(name: str, output_size: tuple[int, int]) -> torch.nn.Module:
    if name == "coord_unet":
        return CoordResizeUNetDBF(output_size=output_size)
    if name == "coord_unet_reflect":
        return CoordResizeUNetDBFReflectionPad(output_size=output_size)
    if name == "coord_local_global_reflect":
        return CoordLocalGlobalUNetDBFReflectionPad(output_size=output_size)
    raise ValueError(f"Unsupported model: {name}")


def complex_from_2ch(tensor: torch.Tensor) -> torch.Tensor:
    return torch.complex(tensor[:, 0], tensor[:, 1])


def summarize_output_size(dataset: ToyDASIQDataset) -> tuple[int, int]:
    item = dataset[0]
    return int(item["y"].shape[1]), int(item["y"].shape[2])


def compute_metrics(y_pred: torch.Tensor, y_target: torch.Tensor) -> dict[str, float]:
    zero = torch.zeros_like(y_target)
    pred_c = complex_from_2ch(y_pred)
    target_c = complex_from_2ch(y_target)
    zero_c = complex_from_2ch(zero)

    pred_mag = torch.abs(pred_c)
    target_mag = torch.abs(target_c)
    zero_mag = torch.abs(zero_c)

    l1_total = torch.mean(torch.abs(y_pred - y_target))
    mse_total = torch.mean((y_pred - y_target) ** 2)
    mae_mag = torch.mean(torch.abs(pred_mag - target_mag))

    l1_zero = torch.mean(torch.abs(zero - y_target))
    mse_zero = torch.mean((zero - y_target) ** 2)
    mae_mag_zero = torch.mean(torch.abs(zero_mag - target_mag))

    return {
        "l1_total": float(l1_total.item()),
        "mse_total": float(mse_total.item()),
        "l1_real": float(torch.mean(torch.abs(y_pred[:, 0] - y_target[:, 0])).item()),
        "l1_imag": float(torch.mean(torch.abs(y_pred[:, 1] - y_target[:, 1])).item()),
        "mse_real": float(torch.mean((y_pred[:, 0] - y_target[:, 0]) ** 2).item()),
        "mse_imag": float(torch.mean((y_pred[:, 1] - y_target[:, 1]) ** 2).item()),
        "mae_magnitude": float(mae_mag.item()),
        "rms_pred_complex": float(torch.sqrt(torch.mean(pred_mag ** 2)).item()),
        "rms_target_complex": float(torch.sqrt(torch.mean(target_mag ** 2)).item()),
        "max_abs_pred_complex": float(torch.max(pred_mag).item()),
        "max_abs_target_complex": float(torch.max(target_mag).item()),
        "l1_zero": float(l1_zero.item()),
        "mse_zero": float(mse_zero.item()),
        "mae_magnitude_zero": float(mae_mag_zero.item()),
        "l1_improvement_vs_zero_percent": float((1.0 - l1_total / l1_zero).item() * 100.0),
        "mse_improvement_vs_zero_percent": float((1.0 - mse_total / mse_zero).item() * 100.0),
        "mae_magnitude_improvement_vs_zero_percent": float((1.0 - mae_mag / mae_mag_zero).item() * 100.0),
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def save_loss_curve(losses: list[float], out_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE loss")
    plt.title("Overfit-all CoordResizeUNetDBF loss curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_metrics_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'sample':38s} {'L1':>10s} {'MSE':>10s} {'MAE_mag':>10s} "
        f"{'rms_pred':>10s} {'rms_tgt':>10s} {'L1_imp%':>10s} {'MSE_imp%':>10s} {'MAE_imp%':>10s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['file_name']:38s} "
            f"{row['l1_total']:10.6f} {row['mse_total']:10.6f} {row['mae_magnitude']:10.6f} "
            f"{row['rms_pred_complex']:10.6f} {row['rms_target_complex']:10.6f} "
            f"{row['l1_improvement_vs_zero_percent']:10.2f} "
            f"{row['mse_improvement_vs_zero_percent']:10.2f} "
            f"{row['mae_magnitude_improvement_vs_zero_percent']:10.2f}"
        )


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "beamformed" / "toy_fieldii"
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyDASIQDataset(data_dir)
    if len(dataset) != 5:
        print(f"Warning: expected 5 samples, found {len(dataset)}")

    sample_names = [dataset[i]["metadata"]["file_name"] for i in range(len(dataset))]
    coord_sources = [dataset[i]["metadata"].get("coord_source", "unknown") for i in range(len(dataset))]
    h, w = summarize_output_size(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dbf_collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, output_size=(h, w)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    best_path = out_dir / "model_best.pt"
    final_path = out_dir / "model_final.pt"
    losses: list[float] = []

    config = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "grad_clip": args.grad_clip,
        "loss_type": args.loss_type,
        "loss": "MSELoss on [B,2,H,W] real/imag; B-mode is visualization only",
        "model": args.model,
        "dataset": "ToyDASIQDataset",
        "collate_fn": "dbf_collate_fn",
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "num_samples": len(dataset),
        "sample_files": sample_names,
        "coord_sources": coord_sources,
        "output_size": [h, w],
        "device": str(device),
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Training {args.model} on {len(dataset)} toy samples")
    print(f"device={device}, batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}, grad_clip={args.grad_clip}")
    print(f"coord sources: {coord_sources}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in loader:
            x = batch["x"].float().to(device)
            y = batch["y"].float().to(device)
            coords_key = "coords_local_global" if args.model == "coord_local_global_reflect" else "coords"
            coords = batch.get(coords_key)
            if coords is not None:
                coords = coords.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x, coords=coords)
            loss = criterion(y_pred, y)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        losses.append(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "output_size": model.output_size,
                "config": config,
            }, best_path)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d}/{args.epochs} | train_loss={epoch_loss:.6f} | best={best_loss:.6f} @ {best_epoch}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "final_loss": losses[-1],
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "output_size": model.output_size,
        "config": config,
    }, final_path)

    save_loss_curve(losses, out_dir / "loss_curve.png")

    model.eval()
    metrics_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_loader = DataLoader(Subset(dataset, [idx]), batch_size=1, shuffle=False, collate_fn=dbf_collate_fn)
            batch = next(iter(sample_loader))
            x = batch["x"].float().to(device)
            y = batch["y"].float().to(device)
            coords_key = "coords_local_global" if args.model == "coord_local_global_reflect" else "coords"
            coords = batch.get(coords_key)
            if coords is not None:
                coords = coords.float().to(device)
            y_pred = model(x, coords=coords)

            file_name = sample["metadata"]["file_name"]
            stem = Path(file_name).stem
            metrics = compute_metrics(y_pred, y)
            figure_path = out_dir / f"{stem}_prediction_vs_target_bmode.png"
            prediction_path = out_dir / f"{stem}_prediction_final.npz"
            target_path = out_dir / f"{stem}_target_final.npz"

            save_prediction_comparison(
                y_pred.squeeze(0),
                y.squeeze(0),
                figure_path,
                title=f"Prediction vs target: {file_name}",
                x_axis_mm=batch.get("x_axis_mm", None),
                z_axis_mm=batch.get("z_axis_mm", None),
            )
            np.savez_compressed(prediction_path, y_pred=y_pred.detach().cpu().numpy())
            np.savez_compressed(target_path, y=y.detach().cpu().numpy())

            metrics_rows.append({
                "file_name": file_name,
                "coord_source": sample["metadata"].get("coord_source", "unknown"),
                "prediction_path": str(prediction_path),
                "target_path": str(target_path),
                "figure_path": str(figure_path),
                **metrics,
            })

    global_metrics = {
        "mean_l1": mean_metric(metrics_rows, "l1_total"),
        "mean_mse": mean_metric(metrics_rows, "mse_total"),
        "mean_mae_magnitude": mean_metric(metrics_rows, "mae_magnitude"),
        "mean_l1_improvement_vs_zero_percent": mean_metric(metrics_rows, "l1_improvement_vs_zero_percent"),
        "mean_mse_improvement_vs_zero_percent": mean_metric(metrics_rows, "mse_improvement_vs_zero_percent"),
        "mean_mae_magnitude_improvement_vs_zero_percent": mean_metric(metrics_rows, "mae_magnitude_improvement_vs_zero_percent"),
        "best_epoch": int(best_epoch),
        "best_loss": float(best_loss),
        "final_loss": float(losses[-1]),
    }

    csv_path = out_dir / "metrics_per_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    metrics_final = {
        "global": global_metrics,
        "per_sample": metrics_rows,
    }
    json_path = out_dir / "metrics_final.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_final, f, indent=2)

    print_metrics_table(metrics_rows)
    print("Global metrics:")
    for key, value in global_metrics.items():
        print(f"{key}: {value}")
    print(f"saved checkpoint: {final_path}")
    print(f"saved best checkpoint: {best_path}")
    print(f"saved config: {out_dir / 'config.json'}")
    print(f"saved metrics: {json_path}")
    print(f"saved metrics csv: {csv_path}")
    print(f"saved loss curve: {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
