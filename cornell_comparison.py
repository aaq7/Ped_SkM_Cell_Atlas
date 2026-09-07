"""
Compares BCH satellite cell subclusters to the major MuSC/progenitor states
identified in the Cornell pediatric muscle atlas (Orton et al.).
"""

import pandas as pd
import scanpy as sc

from utils import (
    CORNELL_DOTPLOT_CMAP,
    CORNELL_FIGURE_GENES,
    H5AD_MRVI,
    LEIDEN_KEY,
    configure_plotting,
    present_genes,
    savefig,
)


def main():
    configure_plotting()

    adata = sc.read_h5ad(H5AD_MRVI)

    cluster_order = sorted(adata.obs[LEIDEN_KEY].unique(), key=int)
    adata.obs["cluster_label"] = pd.Categorical(
        "cluster " + adata.obs[LEIDEN_KEY].astype(str),
        categories=[f"cluster {c}" for c in cluster_order],
    )

    # Dotplot using Cornell's gene list
    genes = present_genes(adata, CORNELL_FIGURE_GENES)
    sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby="cluster_label",
        standard_scale="var",
        cmap=CORNELL_DOTPLOT_CMAP,
        swap_axes=True,
        dendrogram=False,
        show=False,
    )
    savefig("07_cornell_marker_dotplot.png")


if __name__ == "__main__":
    main()
