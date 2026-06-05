from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "dbf_baseline" / "overfit_one"


def to_complex(arr):
    return arr[0] + 1j * arr[1]


def to_db_shared(arr, ref):
    mag = np.abs(arr)
    mag_db = 20.0 * np.log10((mag / (ref + np.finfo(np.float32).eps)) + np.finfo(np.float32).eps)
    return np.clip(mag_db, -60.0, 0.0)


def to_db_local(arr):
    mag = np.abs(arr)
    mag_db = 20.0 * np.log10(mag + np.finfo(np.float32).eps)
    mag_db = mag_db - np.max(mag_db)
    return np.clip(mag_db, -60.0, 0.0)


def save_shared_ref_figure(target_db, pred_db, diff_db, sample_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("W")
        ax.set_ylabel("H")

    im0 = axes[0].imshow(target_db, cmap="gray", vmin=-60, vmax=0, aspect="auto")
    axes[0].set_title("Target DAS-IQ (shared ref, dB)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_db, cmap="gray", vmin=-60, vmax=0, aspect="auto")
    axes[1].set_title("Model output (shared ref, dB)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vmax = float(np.percentile(diff_db, 99)) if np.isfinite(diff_db).any() else 1.0
    vmax = max(vmax, 1e-6)
    im2 = axes[2].imshow(diff_db, cmap="magma", vmin=0, vmax=vmax, aspect="auto")
    axes[2].set_title("|Target - Output| (dB)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Overfit-one comparison (shared reference): {sample_name}", fontsize=11)
    out_path = OUT_DIR / "das_vs_model_shared_ref.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def save_local_bmode_figure(target_db_local, pred_db_local, sample_name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("W")
        ax.set_ylabel("H")

    im0 = axes[0].imshow(target_db_local, cmap="gray", vmin=-60, vmax=0, aspect="auto")
    axes[0].set_title("Target DAS-IQ (local max, dB)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(pred_db_local, cmap="gray", vmin=-60, vmax=0, aspect="auto")
    axes[1].set_title("Model output (local max, dB)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Overfit-one B-mode view (local max): {sample_name}", fontsize=11)
    out_path = OUT_DIR / "das_vs_model_local_bmode.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path




def save_linear_ri_figure(target, pred, sample_name):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    panels = [
        (axes[0, 0], target[0], "Target real"),
        (axes[0, 1], target[1], "Target imag"),
        (axes[0, 2], np.abs(target[0] + 1j * target[1]), "Target |z|"),
        (axes[1, 0], pred[0], "Output real"),
        (axes[1, 1], pred[1], "Output imag"),
        (axes[1, 2], np.abs(pred[0] + 1j * pred[1]), "Output |z|"),
    ]

    for ax, data, title in panels:
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("W")
        ax.set_ylabel("H")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Overfit-one linear RI view: {sample_name}", fontsize=11)
    out_path = OUT_DIR / "das_vs_model_linear_ri.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

def main():
    pred = np.load(OUT_DIR / "prediction_final.npz")["y_pred"]
    target = np.load(OUT_DIR / "target_final.npz")["y"]
    with open(OUT_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pred = np.squeeze(pred)
    target = np.squeeze(target)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, target {target.shape}")
    if pred.ndim != 3 or pred.shape[0] != 2:
        raise ValueError(f"expected [2,H,W] after squeeze, got {pred.shape}")

    target_c = to_complex(target)
    pred_c = to_complex(pred)
    ref = max(np.abs(target_c).max(), np.abs(pred_c).max())

    target_db_shared = to_db_shared(target_c, ref)
    pred_db_shared = to_db_shared(pred_c, ref)
    diff_db = np.abs(target_db_shared - pred_db_shared)

    target_db_local = to_db_local(target_c)
    pred_db_local = to_db_local(pred_c)

    shared_path = save_shared_ref_figure(target_db_shared, pred_db_shared, diff_db, cfg.get('sample_file', 'unknown'))
    local_path = save_local_bmode_figure(target_db_local, pred_db_local, cfg.get('sample_file', 'unknown'))
    linear_path = save_linear_ri_figure(target, pred, cfg.get('sample_file', 'unknown'))

    print(shared_path)
    print(local_path)
    print(linear_path)
    print(f"shared_ref target range: {target_db_shared.min():.2f} to {target_db_shared.max():.2f} dB")
    print(f"shared_ref pred range:   {pred_db_shared.min():.2f} to {pred_db_shared.max():.2f} dB")
    print(f"local_bmode target range: {target_db_local.min():.2f} to {target_db_local.max():.2f} dB")
    print(f"local_bmode pred range:   {pred_db_local.min():.2f} to {pred_db_local.max():.2f} dB")
    print(f"mean abs diff (shared ref): {diff_db.mean():.4f} dB")
    print(f"max abs diff (shared ref):  {diff_db.max():.4f} dB")


if __name__ == "__main__":
    main()
