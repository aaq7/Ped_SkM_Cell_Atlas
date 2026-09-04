Pediatric Muscle Satellite Cell Atlas

Single-cell RNA-seq analysis of satellite cells (MuSCs) from 32 BCH samples. Harmony UMAP clustering was performed on the raw data, creating a Seurat object which was annotated and used for this analysis. MrVI was used to build a sample-corrected latent space and subcluster the cells. Subclusters were characterized via differential expression testing, and the results were compared to the findings from the Cornell pediatric multiome muscle atlas. Lastly, CellRank2 was used to establish a developmental trajectory from quiescence to differentiation.

Pipeline:
seurat_export.Rmd - takes the annotated 32-sample Seruat object, and exports the expression matrix, cell metadata, gene metadata, and cell barcodes.
mrvi_subcluster.py — builds the AnnData object from the Seurat export, trains MrVI on raw counts, and clusters the resulting sample-corrected latent space with the Leiden algorithm.
mrvi_plots.py — visualizes the subclusters and runs differential expression via Wilcoxon test to identify characteristic genes in each subcluster.
cornell_comparison.py — cross-references resolved subclusters against the marker gene sets used in the Cornell pediatric multiome atlas.
cellrank_trajectory.py — roots diffusion pseudotime at the highest-confidence quiescent cell (via high PAX7 expression) and builds a CellRank2 transition model to construct a directional differentiation trajectory.
