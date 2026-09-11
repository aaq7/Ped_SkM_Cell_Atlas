"""
Compares BCH satellite cell subclusters to the major MuSC/progenitor states
identified in the Cornell pediatric muscle atlas (Orton et al.).
"""

import scanpy as sc

from utils import (
    ANNOTATION_KEY,
    CORNELL_DOTPLOT_CMAP,
    CORNELL_FIGURE_GENES,
    H5AD_MRVI,
    annotate_res03,
    configure_plotting,
    present_genes,
    savefig,
)


def main():
    configure_plotting()

    adata = sc.read_h5ad(H5AD_MRVI)
    annotate_res03(adata)

    # Dotplot using Cornell's gene list
    genes = present_genes(adata, CORNELL_FIGURE_GENES)
    sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=ANNOTATION_KEY,
        standard_scale="var",
        cmap=CORNELL_DOTPLOT_CMAP,
        swap_axes=True,
        dendrogram=False,
        show=False,
    )
    savefig("05_cornell_marker_dotplot.png")


if __name__ == "__main__":
    main()
