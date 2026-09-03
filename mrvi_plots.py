"""
Plots the MrVI subclustering.
"""

import scanpy as sc

from utils import (
    H5AD_MRVI,
    LEIDEN_KEY,
    LEIDEN_RESOLUTION,
    MYO_MARKERS,
    N_TOP_DE_GENES,
    SEURAT_HEATMAP_CMAP,
    configure_plotting,
    present_genes,
    savecsv,
    savefig,
)


def find_de_genes(adata, groupby, n_genes):
    """Runs rank_genes_groups (scanpy's equivalent of Seurat's FindAllMarkers)
    and returns the top n_genes markers per cluster as a tidy dataframe."""

    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")

    de = sc.get.rank_genes_groups_df(adata, group=None)
    de = de.groupby("group", sort=False).head(n_genes).reset_index(drop=True)
    de = de.rename(columns={
        "group": "cluster",
        "names": "gene",
        "scores": "score",
        "logfoldchanges": "log2fc",
        "pvals": "pval",
        "pvals_adj": "pval_adj",
    })
    return de[["cluster", "gene", "score", "log2fc", "pval", "pval_adj"]]


def main():
    configure_plotting()

    # load saved h5ad object
    adata = sc.read_h5ad(H5AD_MRVI)

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

    # Differential expression per cluster, to characterize each cluster by its top markers
    de_genes = find_de_genes(adata, LEIDEN_KEY, N_TOP_DE_GENES)
    savecsv(de_genes, "de_genes_per_cluster.csv")

    # Heatmap of the top DE genes per cluster, Seurat DoHeatmap-style purple/black/yellow scale
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=N_TOP_DE_GENES,
        groupby=LEIDEN_KEY,
        standard_scale="var",
        swap_axes=True,
        cmap=SEURAT_HEATMAP_CMAP,
        dendrogram=False,
        show=False,
    )
    savefig("02_de_genes_heatmap.png")

    # Verify the DE test against the myo markers: myo markers + top 3 DE genes per cluster
    # (skip genes already among the myo markers so each cluster contributes new genes)
    top_de_genes = (
        de_genes[~de_genes["gene"].isin(MYO_MARKERS)]
        .groupby("cluster", sort=False)
        .head(3)["gene"]
        .drop_duplicates()
        .tolist()
    )
    myo_markers = present_genes(adata, MYO_MARKERS)
    genes = present_genes(adata, MYO_MARKERS + top_de_genes)

    # Plot dotplot of the selected genes per cluster
    sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=LEIDEN_KEY,
        standard_scale="var",
        cmap="Reds",
        dendrogram=False,
        show=False,
    )
    savefig("03_dotplot_markers.png")

    # Feature plot of the myo markers
    sc.pl.umap(adata, color=myo_markers, ncols=3, cmap="viridis", show=False)
    savefig("04_featureplot_myomarkers.png")

    # Feature plot of the top DE genes per cluster
    sc.pl.umap(adata, color=top_de_genes, ncols=3, cmap="viridis", show=False)
    savefig("05_featureplot_de_genes.png")


if __name__ == "__main__":
    main()
