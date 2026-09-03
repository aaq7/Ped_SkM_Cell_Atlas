"""
Trajectory inference with CellRank 2.3.2

DPT (diffusion pseudotime) rooted at a high-PAX7 quiescent cell drives the PseudotimeKernel.

CONCERN: cellrank2 predicts how cells change and develop over time, identifies cell states, and driver genes. 
It does this via modeling cellular dynamics as a markov chain over a nearest-neighbor graph. As such,
it is not inherently a curve-fitting package, it depends on Kernels to provide biological directionality.
The pseudotime kernel biases nearest-neighbor graph edges towards devlopmental progression. This requires
an explicit progression denoted by an external pseudotime algorithm, usually DPT or Palantir. Palantir works
best for data with several plausible terminal states/branches. DPT works well for trajectories with a clear 
root cell. Since we are interested in differentiation from a quiescent state denoted by PAX7, DPT is a
more suitable choice, although outdated (2016). Additionally, other kernals like CytoTRACE does not work
for our data because it was validated on systems with a wide range in transcriptional complexity. I tried
it and it was not able to produce meaningful results. 

Pipeline:
1) Find a root to anchor pseudotime 
2) Compute DPT pseudotime from the root
3) Build the PseudotimeKernel based on DPT ordering
"""

import numpy as np
import scanpy as sc

from cellrank.kernels import PseudotimeKernel
from cellrank.kernels import ConnectivityKernel

from utils import (
    H5AD_CELLRANK,
    H5AD_MRVI,
    LEIDEN_KEY,
    SEED,
    KERNEL_WEIGHT_CONNECTIVITY,
    KERNEL_WEIGHT_DIRECTIONAL,
    present_genes,
    configure_plotting,
    savefig,
)


# To find a root we need to evaluate expression of myo markers (PAX7)
def get_expression(adata, gene):
    """Returns a gene's log-normalized expression."""

    expr = adata[:, gene].layers["lognorm"]
    return np.asarray(expr.todense() if hasattr(expr, "todense") else expr).ravel()


def find_root(adata):
    """Index of highest PAX7 expressing cell."""

    # Get gene expression array 
    pax7 = get_expression(adata, "PAX7")

    # Computes total UMI per cell and stores in array
    total_counts = np.asarray(adata.layers["counts"].sum(axis=1)).ravel()
    # Filters for where cell total counts exceeds median total counts of a cell
    # Ensures root is from a cell of higher quality 
    reliable = total_counts >= np.median(total_counts)
    # Finds maximum of PAX7 expression among reliable cells --> set as root
    return int(np.flatnonzero(reliable)[np.argmax(pax7[reliable])])


def compute_dpt(adata, root):
    """DPT pseudotime at specified root."""

    # Sets iroot key to the provided root for function input
    adata.uns["iroot"] = root
    # Runs Scanpy's diffusion map algorithm
    sc.tl.diffmap(adata)
    # Calculates diffusion pseudotime
    sc.tl.dpt(adata)

    # Clamp infinity to maximal vals found in pseudotime 
    pt = adata.obs["dpt_pseudotime"] # Extract calculated dpt pseudotime
    n_inf = int(np.isinf(pt).sum())
    if n_inf:
        # replace infinite val with max pseudotime calculated
        adata.obs["dpt_pseudotime"] = pt.replace(np.inf, pt[np.isfinite(pt)].max())

    return adata


def main():
    configure_plotting()
    np.random.seed(SEED)

    # Load the saved MrVI dataset
    adata = sc.read_h5ad(H5AD_MRVI)
    # Set annData object to lognormalized expression (needed for Scanpy DPT)
    adata.X = adata.layers["lognorm"].copy()

    # Find the root cell based on highest PAX7 expression
    root = find_root(adata)
    # Compute DPT pseudotime from the root
    adata = compute_dpt(adata, root)

    # Build the PseudotimeKernel based on the computed DPT pseudotime
    pk = PseudotimeKernel(adata, time_key="dpt_pseudotime").compute_transition_matrix(threshold_scheme="soft")

    # Visualize the DPT pseudotime on the UMAP embedding
    sc.pl.umap(adata, color="dpt_pseudotime", cmap="plasma", show=False)
    savefig("04_cellrank_pseudotime.png")


    # Additional plot from Figure 2 of Theis et al. 2024

    # Build connectivity kernel and combine with directional pseudotime 
    ck = ConnectivityKernel(adata).compute_transition_matrix()
    combined_kernel = (
        KERNEL_WEIGHT_DIRECTIONAL * pk + KERNEL_WEIGHT_CONNECTIVITY * ck
    )

    # Project transition streamlines onto the UMAP
    combined_kernel.plot_projection(
        basis="umap", color=LEIDEN_KEY, legend_loc="right", recompute=True, show=False
    )
    savefig("05_cellrank_streamlines.png")

    # Save final AnnData object
    adata.write_h5ad(H5AD_CELLRANK)

if __name__ == "__main__":
    main()