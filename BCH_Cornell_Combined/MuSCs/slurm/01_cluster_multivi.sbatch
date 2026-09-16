#!/bin/bash
# CPU-only: subset Mingke's MultiVI object to MuSCs, cluster on its latent, and
# regenerate the non-age-model figures/comparisons against it.
#
# Submit from the MuSCs/ directory: sbatch slurm/01_cluster_multivi.sbatch
#SBATCH -J musc_multivi_cluster
#SBATCH -A bch
#SBATCH -p bch-compute
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
# Point this at wherever you copied multivi_results.h5ad (from the
# Cornell_BCH_orig_ident SharePoint bundle) on E3 storage, if it isn't
# already at the default path utils.py expects (next to this script).
: "${MULTIVI_SOURCE_H5AD:=$PWD/multivi_results.h5ad}"
export MULTIVI_SOURCE_H5AD

python multivi_subcluster.py
python mrvi_plots.py
python cornell_comparison.py
python pseudotimekernel_trajectory.py
