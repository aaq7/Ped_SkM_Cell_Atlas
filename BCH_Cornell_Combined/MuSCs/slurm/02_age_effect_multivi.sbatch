#!/bin/bash
# GPU: trains age_characterization.py's scvi.external.MRVI age-effect model
# (still MrVI-on-RNA-counts even in the multivi variant -- see the caveat in
# utils.py next to AGE_MODEL_DIR) against the MultiVI-clustered object.
#
# Run 01_cluster_multivi.sbatch first. Submit from the MuSCs/ directory:
# sbatch slurm/02_age_effect_multivi.sbatch
#SBATCH -J musc_multivi_age
#SBATCH -A bch
#SBATCH -p bch-gpu
#SBATCH --gpus 1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module load miniforge
conda activate musc-multivi

export PIPELINE_SOURCE=multivi

python age_characterization.py
