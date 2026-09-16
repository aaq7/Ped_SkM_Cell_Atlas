"""
Plots the MrVI subclustering.
"""

import scanpy as sc

from utils import (
    BATCH_KEY,
    CELLTYPE_KEY,
    H5AD_FULL_ATLAS,
    H5AD_ACTIVE,
    LEIDEN_KEY,
    LEIDEN_RESOLUTION,
    MYO_MARKERS,
    configure_plotting,
    present_genes,
    savefig,
)


def plot_full_atlas_umap():
    """Full-atlas UMAP (all cell types, pre-MuSC-subset), using the atlas's own precomputed
    embedding -- sanity check that celltype populations are coherent and well-mixed across
    the BCH/Cornell cohorts before we subset down to MuSCs."""

    # Backed mode: only obs/obsm are needed, not the (12.9GB) expression matrix
    adata = sc.read_h5ad(H5AD_FULL_ATLAS, backed="r")

    sc.pl.embedding(
        adata,
        basis="X_umap",
        color=[CELLTYPE_KEY, BATCH_KEY],
        frameon=False,
        title=["celltype", "batch (B=BCH, C=Cornell)"],
        show=False,
    )
    savefig("00_full_atlas_umap.png")


def main():
    configure_plotting()

    # Full-atlas UMAP
    plot_full_atlas_umap()

    # load saved h5ad object
    adata = sc.read_h5ad(H5AD_ACTIVE)

    # Plot UMAP and leiden subclusters
    sc.pl.umap(
        adata,
        color=LEIDEN_KEY,
        legend_loc="right margin",
        frameon=False,
        title=f"MrVI Leiden clusters, resolution {LEIDEN_RESOLUTION}",
        show=False,
    )
    savefig("01_umap_subclusters.png")

    # UMAP colored by cohort/assay 
    sc.pl.umap(
        adata,
        color=BATCH_KEY,
        legend_loc="right margin",
        frameon=False,
        title="batch (B=BCH, C=Cornell)",
        show=False,
    )
    savefig("01b_umap_subclusters_by_batch.png")

    # Feature plot of the myo markers
    myo_markers = present_genes(adata, MYO_MARKERS)
    sc.pl.umap(adata, color=myo_markers, ncols=3, cmap="viridis", show=False)
    savefig("04_featureplot_myomarkers.png")


if __name__ == "__main__":
    main()
