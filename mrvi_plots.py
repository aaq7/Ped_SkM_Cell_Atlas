"""
Plots the MrVI subclustering.
"""

import scanpy as sc

from utils import (
    H5AD_MRVI,
    LEIDEN_KEY,
    LEIDEN_RESOLUTION,
    MYO_MARKERS,
    TARGET_GENES,
    configure_plotting,
    present_genes,
    savefig,
)


def main():
    configure_plotting()

    # load saved h5ad object
    adata = sc.read_h5ad(H5AD_MRVI)

    # Plot UMAP and leiden subclusters
    sc.pl.umap(adata, color=LEIDEN_KEY, palette="tab20", legend_loc="on data",
              title=f"MrVI Leiden clusters, resolution {LEIDEN_RESOLUTION}")
    savefig("01_umap_subclusters.png")

    # In case the filtering dropped our targets, ensure myo marker, SELENON, ELL2 are present
    genes = present_genes(adata, MYO_MARKERS + TARGET_GENES)

    # Plot dotplot of the selected genes per cluster
    sc.pl.dotplot(adata, var_names=genes, groupby=LEIDEN_KEY,
                  standard_scale="var", cmap="Reds")
    savefig("02_dotplot_markers.png")

    # Feature plot of the target genes 
    sc.pl.umap(adata, color=genes, ncols=3, cmap="viridis")
    savefig("03_featureplot_targets.png")


if __name__ == "__main__":
    main()
