#!/usr/bin/env bash
#SBATCH --job-name=toy_fieldii
#SBATCH --partition=thinkstation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
SIM_DIR="${REPO_ROOT}/simulations/fieldii"

mkdir -p "${REPO_ROOT}/logs/slurm"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-local}"
echo "Repo: ${REPO_ROOT}"
echo "Simulation dir: ${SIM_DIR}"
echo "Started at: $(date)"

cd "${SIM_DIR}"

if [[ -n "${MATLAB_BIN:-}" ]]; then
    MATLAB_CMD="${MATLAB_BIN}"
elif command -v matlab >/dev/null 2>&1; then
    MATLAB_CMD="$(command -v matlab)"
else
    echo "ERROR: MATLAB was not found on PATH."
    echo "Set MATLAB_BIN=/path/to/matlab or load MATLAB before submitting."
    exit 127
fi

"${MATLAB_CMD}" -batch "simulate_toy_dataset"

echo "Finished at: $(date)"
