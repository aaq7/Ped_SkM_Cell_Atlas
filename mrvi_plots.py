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
    """Runs rank_genes_groups and returns the top n_genes markers per cluster."""

    # Run with wilcoxon test for statistic
    sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")

    # Get results in dataframe
    de = sc.get.rank_genes_groups_df(adata, group=None)
    # Group by cluster and get the top n genes
    de = de.groupby("group", sort=False).head(n_genes).reset_index(drop=True)

    # Return cols of the dataframe
    return de["group", "names", "scores", "logfoldchanges", "pvals", "pvals_adj"]


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

    # Select the top 3 DE genes per cluster
    top_de_genes = (
        de_genes[~de_genes["names"].isin(MYO_MARKERS)] # skip if myo marker
        .groupby("group", sort=False) # group by cluster
        .head(3)["names"] # gene names 
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
