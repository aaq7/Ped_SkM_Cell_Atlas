"""
Trajectory inference with CellRank 2.3.2 -- CytoTRACEKernel

CytoTRACE kernel is fundamentally built on gene sequencing depth and transcriptional diversity. The assumption is that
cells with more gene sequencing depth are more stem-cell like, and that cells with less gene sequencing depth are more
differentiated. 

However, what I am seeing is that cluster 6 and 7 keeps coming up as further down pseudotime, even though their gene
expression profiles would suggest otherwise. This is a limitation of the CytoTRACE kernel.

https://rnabio.org/module-08-scrna/0008/07/01/Differentiation_trajectory/
"""

import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

from cellrank.kernels import CytoTRACEKernel, ConnectivityKernel
from cellrank.estimators import GPCCA

from utils import (
    H5AD_CYTOTRACEKERNEL,
    H5AD_MRVI,
    LEIDEN_KEY,
    SEED,
    KERNEL_WEIGHT_CONNECTIVITY,
    KERNEL_WEIGHT_DIRECTIONAL,
    GPCCA_N_COMPONENTS,
    GPCCA_N_STATES,
    GPCCA_N_TERMINAL_STATES,
    FIG_DIR,
    configure_plotting,
    savefig,
)


def compute_cytotrace_kernel(adata):
    """Builds the CytoTRACEKernel from the CytoTRACE score."""

    ctk = CytoTRACEKernel(adata).compute_cytotrace(layer="X")
    return ctk.compute_transition_matrix(threshold_scheme="soft")


def fit_terminal_states(combined_kernel):
    """Fits a GPCCA estimator and identifies terminal states/fate probabilities."""

    g = GPCCA(combined_kernel)
    # Schur decomposition of the transition matrix
    # Factorizes any square matrix into a unitary matrix and an upper triangular matrix
    g.compute_schur(n_components=GPCCA_N_COMPONENTS)
    # Coarse-grain the chain into macrostates
    g.compute_macrostates(n_states=GPCCA_N_STATES, cluster_key=LEIDEN_KEY)
    # method="top_n" picks a fixed number of terminal states
    g.predict_terminal_states(method="top_n", n_states=GPCCA_N_TERMINAL_STATES)
    g.compute_fate_probabilities()
    return g


def plot_depth_by_cluster(adata, output_path):
    """QC violins (sequencing depth, gene diversity, mitochondrial%) per Leiden cluster."""

    qc_cols = ["nCount_RNA", "nFeature_RNA", "percent.mt"]
    fig, axes = plt.subplots(1, len(qc_cols), figsize=(4 * len(qc_cols), 4))
    for ax, col in zip(axes, qc_cols):
        sc.pl.violin(adata, keys=col, groupby=LEIDEN_KEY, ax=ax, show=False, rotation=90)
        ax.set_title(col)
    fig.tight_layout()
    savefig(output_path, fig)

def main():
    configure_plotting()
    np.random.seed(SEED)

    # Load the saved MrVI dataset
    adata = sc.read_h5ad(H5AD_MRVI)
    # Set annData object to lognormalized expression (needed for CytoTRACE gene-expressed counts)
    adata.X = adata.layers["lognorm"].copy()

    # Build the CytoTRACEKernel based on the computed CytoTRACE pseudotime
    ctk = compute_cytotrace_kernel(adata)

    # Visualize the CytoTRACE pseudotime on the UMAP embedding
    sc.pl.umap(adata, color="ct_pseudotime", cmap="plasma", show=False)
    savefig("08_cytotracekernel_umap.png")

    # Run QC violins (sequencing depth, gene diversity, mitochondrial%) per Leiden cluster
    plot_depth_by_cluster(adata, FIG_DIR / "08b_cytotracekernel_qc_by_cluster.png")

    # Build connectivity kernel and combine with directional CytoTRACE pseudotime
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    combined_kernel = (
        KERNEL_WEIGHT_DIRECTIONAL * ctk + KERNEL_WEIGHT_CONNECTIVITY * ck
    )

    # Project transition streamlines onto the UMAP
    combined_kernel.plot_projection(
        basis="umap", color=LEIDEN_KEY, legend_loc="right", recompute=True, show=False
    )
    savefig("09_cytotracekernel_streamlines.png")

    # Fit GPCCA estimator
    g = fit_terminal_states(combined_kernel)

    # Inferred terminal states
    g.plot_macrostates(which="terminal", basis="umap", legend_loc="right", show=False)
    savefig("10_cytotracekernel_terminal_states.png")

    # Fate probabilities per terminal state
    g.plot_fate_probabilities(basis="umap", same_plot=False, show=False)
    savefig("11_cytotracekernel_fate_probability.png")

    # Save final AnnData object
    adata.write_h5ad(H5AD_CYTOTRACEKERNEL)


if __name__ == "__main__":
    main()
