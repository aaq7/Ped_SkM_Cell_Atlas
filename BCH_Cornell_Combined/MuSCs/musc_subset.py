"""
Subsets the combined BCH+Cornell atlas down to satellite cells (MuSCs).
"""

import scanpy as sc

from utils import (
    CELLTYPE_KEY,
    H5AD_FULL_ATLAS,
    H5AD_RAW,
    MUSC_LABEL,
)


def build_anndata():
    """Loads the combined atlas and subsets it to MuSCs."""

    adata = sc.read_h5ad(H5AD_FULL_ATLAS)

    keep = adata.obs[CELLTYPE_KEY].astype(str) == MUSC_LABEL
    adata = adata[keep].copy()

    # Save raw copy of counts for MrVI (X is already raw SoupX-corrected counts)
    adata.layers["counts"] = adata.X.copy()

    adata.write_h5ad(H5AD_RAW)
    return adata


def main():
    adata = build_anndata()
    print(f"Subsetted {adata.n_obs} MuSCs x {adata.n_vars} genes -> {H5AD_RAW}")


if __name__ == "__main__":
    main()
