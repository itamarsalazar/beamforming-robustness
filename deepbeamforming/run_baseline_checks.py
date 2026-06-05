"""Run a complete baseline validation for the deep beamforming pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index for the short overfit run")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for the short overfit run")
    return parser.parse_args()


def run_command(cmd: list[str], cwd: Path) -> tuple[bool, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output


def file_status(path: Path) -> bool:
    return path.exists() and path.is_file()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    status = {
        "dataset": False,
        "model": False,
        "forward_backward": False,
        "short_train": False,
    }
    problems: list[str] = []
    generated_files: list[str] = []

    commands = [
        ("dataset", [py, "deepbeamforming/check_dataset.py"]),
        ("model", [py, "deepbeamforming/check_model.py"]),
        ("short_train", [py, "deepbeamforming/train_overfit_one.py", "--sample-index", str(args.sample_index), "--epochs", str(args.epochs)]),
    ]

    print("Running baseline checks...\n")
    for name, cmd in commands:
        ok, output = run_command(cmd, repo_root)
        status[name] = ok
        print(f"[{name}] {'OK' if ok else 'FAIL'}")
        if output.strip():
            print(output.rstrip())
        print()
        if not ok:
            problems.append(f"{name} check failed")

    out_dir = repo_root / "experiments" / "dbf_baseline" / "overfit_one"
    expected_files = [
        out_dir / "model_final.pt",
        out_dir / "prediction_final.npz",
        out_dir / "target_final.npz",
        out_dir / "config.json",
        out_dir / "loss_curve.png",
        out_dir / "prediction_vs_target_bmode.png",
    ]

    for path in expected_files:
        if file_status(path):
            generated_files.append(str(path))
        else:
            problems.append(f"Missing expected file: {path}")

    status["forward_backward"] = status["model"]

    print("Baseline diagnosis")
    print(f"Dataset ready: {'ready' if status['dataset'] else 'not ready'}")
    print(f"Model ready: {'ready' if status['model'] else 'not ready'}")
    print(f"Forward/backward ready: {'ready' if status['forward_backward'] else 'not ready'}")
    print(f"Short training ready: {'ready' if status['short_train'] else 'not ready'}")
    print("Generated files:")
    if generated_files:
        for item in generated_files:
            print(f"- {item}")
    else:
        print("- none")
    print("Problems detected:")
    if problems:
        for item in problems:
            print(f"- {item}")
    else:
        print("- none")

    if all(status.values()) and not problems:
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
