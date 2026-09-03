"""
Compares BCH satellite cell subclusters to the major MuSC/progenitor states
identified in the Cornell pediatric muscle atlas (Orton et al.).
"""

import scanpy as sc

from utils import (
    CORNELL_MUSC_CLASSES,
    H5AD_MRVI,
    LEIDEN_KEY,
    configure_plotting,
    present_genes,
    savefig,
)


def main():
    configure_plotting()

    adata = sc.read_h5ad(H5AD_MRVI)

    # Dotplot of the Cornell marker genes, grouped by state, to see if our clusters agree
    marker_groups = {name: present_genes(adata, genes) for name, genes in CORNELL_MUSC_CLASSES.items()}
    sc.pl.dotplot(
        adata,
        var_names=marker_groups,
        groupby=LEIDEN_KEY,
        standard_scale="var",
        cmap="Reds",
        dendrogram=False,
        show=False,
    )
    savefig("06_cornell_marker_dotplot.png")


if __name__ == "__main__":
    main()
