"""Overfit one toy DAS-IQ sample to validate the full deep beamforming pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from datasets import ToyDASIQDataset, dbf_collate_fn
    from models import ResizeConvDBF, CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad
    from visualize import save_prediction_comparison
else:  # pragma: no cover
    from .datasets import ToyDASIQDataset, dbf_collate_fn
    from .models import ResizeConvDBF, CoordResizeUNetDBF, CoordResizeUNetDBFReflectionPad, CoordLocalGlobalUNetDBFReflectionPad
    from .visualize import save_prediction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0, help="Index of the sample to overfit")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--model", choices=["resizeconv", "coord_unet", "coord_unet_reflect", "coord_local_global_reflect"], default="resizeconv")
    parser.add_argument("--loss-type", choices=["l1", "mse", "mse_l1", "mse_mag"], default="l1")
    parser.add_argument("--alpha", type=float, default=0.1, help="L1 weight for mse_l1")
    parser.add_argument("--beta", type=float, default=0.1, help="Magnitude L1 weight for mse_mag")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Optional gradient clipping max norm; <=0 disables it")
    parser.add_argument("--output-dir", type=str, default="", help="Optional output directory")
    return parser.parse_args()


def build_model(name: str, output_size: tuple[int, int]) -> torch.nn.Module:
    if name == "resizeconv":
        return ResizeConvDBF(output_size=output_size)
    if name == "coord_unet":
        return CoordResizeUNetDBF(output_size=output_size)
    if name == "coord_unet_reflect":
        return CoordResizeUNetDBFReflectionPad(output_size=output_size)
    if name == "coord_local_global_reflect":
        return CoordLocalGlobalUNetDBFReflectionPad(output_size=output_size)
    raise ValueError(f"Unsupported model: {name}")


def abs_complex(y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(y[:, 0] ** 2 + y[:, 1] ** 2 + torch.finfo(y.dtype).eps)


def compute_loss(y_pred: torch.Tensor, y_target: torch.Tensor, loss_type: str, alpha: float, beta: float) -> torch.Tensor:
    l1 = torch.nn.functional.l1_loss(y_pred, y_target)
    mse = torch.nn.functional.mse_loss(y_pred, y_target)

    if loss_type == "l1":
        return l1
    if loss_type == "mse":
        return mse
    if loss_type == "mse_l1":
        return mse + alpha * l1
    if loss_type == "mse_mag":
        mag_l1 = torch.nn.functional.l1_loss(abs_complex(y_pred), abs_complex(y_target))
        return mse + beta * mag_l1
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def complex_from_2ch(tensor: torch.Tensor) -> torch.Tensor:
    return torch.complex(tensor[:, 0], tensor[:, 1])


def compute_metrics(y_pred: torch.Tensor, y_target: torch.Tensor) -> dict:
    zero = torch.zeros_like(y_target)
    pred_c = complex_from_2ch(y_pred)
    target_c = complex_from_2ch(y_target)
    zero_c = complex_from_2ch(zero)
    pred_mag = torch.abs(pred_c)
    target_mag = torch.abs(target_c)
    zero_mag = torch.abs(zero_c)

    l1_total = torch.mean(torch.abs(y_pred - y_target))
    mse_total = torch.mean((y_pred - y_target) ** 2)
    l1_zero = torch.mean(torch.abs(zero - y_target))
    mse_zero = torch.mean((zero - y_target) ** 2)
    mae_mag = torch.mean(torch.abs(pred_mag - target_mag))
    mae_mag_zero = torch.mean(torch.abs(zero_mag - target_mag))

    return {
        "l1_total": float(l1_total.item()),
        "mse_total": float(mse_total.item()),
        "l1_zero": float(l1_zero.item()),
        "mse_zero": float(mse_zero.item()),
        "mae_magnitude": float(mae_mag.item()),
        "mae_magnitude_zero": float(mae_mag_zero.item()),
        "l1_improvement_vs_zero_percent": float((1.0 - l1_total / l1_zero).item() * 100.0),
        "mse_improvement_vs_zero_percent": float((1.0 - mse_total / mse_zero).item() * 100.0),
        "mae_magnitude_improvement_vs_zero_percent": float((1.0 - mae_mag / mae_mag_zero).item() * 100.0),
        "l1_real": float(torch.mean(torch.abs(y_pred[:, 0] - y_target[:, 0])).item()),
        "l1_imag": float(torch.mean(torch.abs(y_pred[:, 1] - y_target[:, 1])).item()),
        "mse_real": float(torch.mean((y_pred[:, 0] - y_target[:, 0]) ** 2).item()),
        "mse_imag": float(torch.mean((y_pred[:, 1] - y_target[:, 1]) ** 2).item()),
        "rms_pred_complex": float(torch.sqrt(torch.mean(pred_mag ** 2)).item()),
        "rms_target_complex": float(torch.sqrt(torch.mean(target_mag ** 2)).item()),
        "max_abs_pred_complex": float(torch.max(pred_mag).item()),
        "max_abs_target_complex": float(torch.max(target_mag).item()),
    }


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "beamformed" / "toy_fieldii"
    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
    else:
        out_dir = repo_root / "experiments" / "dbf_baseline" / "overfit_one"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ToyDASIQDataset(data_dir)
    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} out of range for dataset of size {len(dataset)}")

    sample = dataset[args.sample_index]
    loader = DataLoader(Subset(dataset, [args.sample_index]), batch_size=1, shuffle=False, collate_fn=dbf_collate_fn)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = batch["x"].float().to(device)
    y = batch["y"].float().to(device)
    coords_key = "coords_local_global" if args.model == "coord_local_global_reflect" else "coords"
    coords = batch.get(coords_key)
    if coords is not None:
        coords = coords.float().to(device)

    model = build_model(args.model, (int(y.shape[2]), int(y.shape[3]))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    print(f"Training on sample: {sample['metadata']['file_name']}")
    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"loss_type: {args.loss_type}, alpha: {args.alpha}, beta: {args.beta}, lr: {args.lr}")
    print(f"grad_clip: {args.grad_clip}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    if coords is not None:
        print(f"coords shape: {tuple(coords.shape)}")
        print(f"coord source: {sample['metadata'].get('coord_source', 'unknown')}")

    best_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = out_dir / "model_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x, coords=coords)
        loss = compute_loss(y_pred, y, args.loss_type, args.alpha, args.beta)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "best_loss": best_loss,
                "output_size": (int(y.shape[2]), int(y.shape[3])),
            }, best_checkpoint_path)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d}/{args.epochs} | loss={loss_value:.6f} | best={best_loss:.6f} @ {best_epoch}")

    model.eval()
    with torch.no_grad():
        y_pred = model(x, coords=coords)

    metrics = compute_metrics(y_pred, y)
    metrics["final_loss"] = float(losses[-1])
    metrics["best_loss"] = float(best_loss)
    metrics["best_epoch"] = int(best_epoch)
    checkpoint_path = out_dir / "model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "output_size": (int(y.shape[2]), int(y.shape[3])),
        "coord_source": sample["metadata"].get("coord_source", "unknown"),
        "metrics": metrics,
    }, checkpoint_path)

    np.savez_compressed(out_dir / "prediction_final.npz", y_pred=y_pred.detach().cpu().numpy())
    np.savez_compressed(out_dir / "target_final.npz", y=y.detach().cpu().numpy())

    config = {
        "sample_index": args.sample_index,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": 1,
        "model": args.model,
        "loss_type": args.loss_type,
        "alpha": args.alpha,
        "beta": args.beta,
        "grad_clip": args.grad_clip,
        "loss": "configured on [B,2,H,W] real/imag; B-mode is visualization only",
        "device": str(device),
        "dataset": "ToyDASIQDataset",
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "sample_file": sample["metadata"]["file_name"],
        "input_shape": list(x.shape),
        "target_shape": list(y.shape),
        "coord_source": sample["metadata"].get("coord_source", "unknown"),
        "metrics": metrics,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "metrics_final.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Overfit-one loss curve ({args.model}, {args.loss_type})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    comparison_path = save_prediction_comparison(
        y_pred.squeeze(0),
        y.squeeze(0),
        out_dir / "prediction_vs_target_bmode.png",
        title=f"Prediction vs target: {sample['metadata']['file_name']} ({args.model}, {args.loss_type})",
        x_axis_mm=batch.get("x_axis_mm", None),
        z_axis_mm=batch.get("z_axis_mm", None),
    )

    print(f"saved comparison: {comparison_path}")
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved best checkpoint: {best_checkpoint_path}")
    print(f"saved config: {out_dir / 'config.json'}")
    print(f"saved metrics: {out_dir / 'metrics_final.json'}")
    print(f"saved prediction: {out_dir / 'prediction_final.npz'}")
    print(f"saved target: {out_dir / 'target_final.npz'}")
    print(f"final loss: {losses[-1]:.6f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")


if __name__ == "__main__":
    main()
