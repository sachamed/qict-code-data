#!/bin/bash
#SBATCH --job-name=qict-c10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/c10-%j.out
#SBATCH --error=logs/c10-%j.err
# Fail-closed preflight only. Real HPC settings follow physical matvec benchmark.
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
cd "${SLURM_SUBMIT_DIR:?}"
python src/stage_guard.py --stage c10
echo 'BLOCKED: physical implementation and independent certificates required' >&2
exit 2
