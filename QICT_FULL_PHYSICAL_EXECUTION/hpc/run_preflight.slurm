#!/bin/bash
#SBATCH --job-name=qict-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/preflight-%j.out
#SBATCH --error=logs/preflight-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
python -m unittest discover -s tests -v
python src/stage_guard.py --stage kato
