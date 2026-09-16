"""
Subclusters the satellite cells using MrVI.

https://docs.scvi-tools.org/en/1.2.2/tutorials/notebooks/scrna/MrVI_tutorial.html
"""

import os

from pathlib import Path

import pandas as pd
import scanpy as sc

from utils import (
    BATCH_KEY,
    H5AD_MRVI,
    H5AD_RAW,
    LEIDEN_KEY,
    LEIDEN_RESOLUTION,
    MRVI_BATCH_SIZE,
    MRVI_MAX_EPOCHS,
    MRVI_MODEL_DIR,
    MRVI_N_LATENT,
    N_NEIGHBORS,
    SAMPLE_KEY,
    SEED,
    configure_plotting,
)


def preprocess(adata):
    """Filters low-detection genes and log-normalizes for MrVI."""

    # Remove genes that are detected in fewer than 5 cells
    sc.pp.filter_genes(adata, min_cells=5)

    # Create copy of UMI counts into lognorm layer
    adata.layers["lognorm"] = adata.layers["counts"].copy()
    # Point .X at the copied lognorm layer to preserve original
    adata.X = adata.layers["lognorm"]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized log-transformed data in the "lognorm" layer
    adata.layers["lognorm"] = adata.X.copy()
    return adata


def load_model_var_names(model_dir):
    """Reads the exact gene set a cached MrVI checkpoint was trained on."""

    import torch

    state = torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=False)
    return pd.Index(state["var_names"])


def train_mrvi(adata, sample_key):
    """Trains MrVI on raw counts."""

    from scvi.external import MRVI

    # Skip retraining on reruns; reuse the cached model's exact trained gene set
    if MRVI_MODEL_DIR.exists():
        model_genes = load_model_var_names(MRVI_MODEL_DIR)
        return MRVI.load(str(MRVI_MODEL_DIR), adata=adata[:, model_genes].copy())

    # MrVI models the raw count distribution directly. batch_key is critical here: this
    # atlas combines two cohorts run on different assays (BCH snRNA-seq vs Cornell
    # snRNA-seq/snATAC-seq multiome). Without telling MrVI about that technical split, the
    # per-sample latent has no way to separate "this sample differs because of donor
    # biology" from "this sample differs because it's a different assay entirely" -- the
    # assay effect dominates and clusters split cleanly by cohort instead of cell state.
    MRVI.setup_anndata(adata, layer="counts", sample_key=sample_key, batch_key=BATCH_KEY)

    model = MRVI(adata, n_latent=MRVI_N_LATENT) # latent dimensionality, recommended 10-30
    model.train(
        max_epochs=MRVI_MAX_EPOCHS, # change this in utils
        batch_size=MRVI_BATCH_SIZE, # 256 cells per batch (optimize memory)
        early_stopping=True, # stop training when performance plataeus
    )

    MRVI_MODEL_DIR.mkdir(parents=True, exist_ok=True) # make directory for future use
    model.save(str(MRVI_MODEL_DIR), overwrite=True)
    return model


def main():
    configure_plotting()
    sc.settings.n_jobs = 4  # 4 cpu cores

    # Load the MuSC-only raw object (from musc_subset.py)
    adata = sc.read_h5ad(H5AD_RAW)

    # Train MrVI on raw counts (reuses the cached model's exact trained gene set if present)
    model = train_mrvi(adata, SAMPLE_KEY)
    # Extracts sample-corrected latent space from mrvi model and stores in annData object
    adata.obsm["X_mrvi_u"] = model.get_latent_representation(give_z=False)

    # Preprocess (filter low-detection genes, log-normalize) for downstream marker/DE analysis
    adata = preprocess(adata)

    # Construct k-nearest neighbor graph on n = 15 neighbors
    sc.pp.neighbors(adata, use_rep="X_mrvi_u", n_neighbors=N_NEIGHBORS,
                    random_state=SEED)
    # Calculate UMAP 2D embedding
    sc.tl.umap(adata, random_state=SEED)

    # Run leiden clustering graph algoroithm
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY,
                 flavor="igraph", n_iterations=2, directed=False,
                 random_state=SEED)
    adata.obs[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].astype(str)  # avoid category dtype issues on read-back

    # Write formatted annData object to h5ad
    adata.write_h5ad(H5AD_MRVI)


if __name__ == "__main__":
    main()
