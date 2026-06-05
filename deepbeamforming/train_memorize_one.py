"""Train a direct memorization model on one toy DAS-IQ sample."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))
    from datasets import ToyDASIQDataset, dbf_collate_fn
    from models import MemorizeImageDBF
    from visualize import save_prediction_comparison
else:  # pragma: no cover
    from .datasets import ToyDASIQDataset, dbf_collate_fn
    from .models import MemorizeImageDBF
    from .visualize import save_prediction_comparison


def complex_from_2ch(tensor: torch.Tensor) -> torch.Tensor:
    return torch.complex(tensor[:, 0], tensor[:, 1])


def compute_metrics(y_pred: torch.Tensor, y_target: torch.Tensor) -> dict:
    pred_c = complex_from_2ch(y_pred)
    target_c = complex_from_2ch(y_target)
    pred_mag = torch.abs(pred_c)
    target_mag = torch.abs(target_c)

    return {
        "l1": float(torch.mean(torch.abs(y_pred - y_target)).item()),
        "mse": float(torch.mean((y_pred - y_target) ** 2).item()),
        "mae_magnitude": float(torch.mean(torch.abs(pred_mag - target_mag)).item()),
        "mean_abs_target_complex": float(torch.mean(target_mag).item()),
        "max_abs_target_complex": float(torch.max(target_mag).item()),
        "rms_target_complex": float(torch.sqrt(torch.mean(target_mag ** 2)).item()),
        "mean_abs_pred_complex": float(torch.mean(pred_mag).item()),
        "max_abs_pred_complex": float(torch.max(pred_mag).item()),
        "rms_pred_complex": float(torch.sqrt(torch.mean(pred_mag ** 2)).item()),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "beamformed" / "toy_fieldii"
    out_dir = repo_root / "experiments" / "dbf_baseline" / "memorize_one"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_index = 1  # sample_002_points
    epochs = 1000
    lr = 1e-2

    dataset = ToyDASIQDataset(data_dir)
    sample = dataset[sample_index]
    loader = DataLoader(Subset(dataset, [sample_index]), batch_size=1, shuffle=False, collate_fn=dbf_collate_fn)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = batch["x"].float().to(device)
    y = batch["y"].float().to(device)

    model = MemorizeImageDBF(output_size=(int(y.shape[2]), int(y.shape[3]))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()

    losses = []
    print(f"Training MemorizeImageDBF on sample: {sample['metadata']['file_name']}")
    print(f"device: {device}")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"epoch {epoch:04d}/{epochs} | loss={loss_value:.8f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(x)

    metrics = compute_metrics(y_pred, y)

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "sample_index": sample_index,
        "sample_file": sample["metadata"]["file_name"],
        "epochs": epochs,
        "learning_rate": lr,
        "output_size": model.output_size,
        "metrics": metrics,
    }, out_dir / "model_final.pt")

    with open(out_dir / "metrics_final.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(losses, linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("L1 loss")
    plt.title("Memorize-one loss curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    comparison_path = save_prediction_comparison(
        y_pred.squeeze(0),
        y.squeeze(0),
        out_dir / "prediction_vs_target_bmode.png",
        title=f"MemorizeImageDBF: {sample['metadata']['file_name']}",
    )

    print(f"saved checkpoint: {out_dir / 'model_final.pt'}")
    print(f"saved loss curve: {out_dir / 'loss_curve.png'}")
    print(f"saved comparison: {comparison_path}")
    print(f"saved metrics: {out_dir / 'metrics_final.json'}")
    print(f"final loss: {losses[-1]:.8f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")


if __name__ == "__main__":
    main()
