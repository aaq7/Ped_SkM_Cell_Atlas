"""
Subclusters the satellite cells using Mingke's MultiVI (RNA+ATAC) object.

Mirrors mrvi_subcluster.py's neighbors/UMAP/leiden steps, but starts from a
MultiVI-integrated object instead of training MrVI ourselves. Run this with
PIPELINE_SOURCE=multivi so every downstream script (age_characterization.py,
cornell_comparison.py, pseudotimekernel_trajectory.py, mrvi_plots.py) reads
the MultiVI outputs instead of the MrVI ones -- see utils.py's H5AD_ACTIVE.

NOTE: Mingke's object's exact schema (obs columns, obsm latent key) has not
been verified yet -- this script checks for what it needs and fails with a
clear message rather than silently assuming column/key names. Update
MULTIVI_SOURCE_H5AD / MULTIVI_LATENT_KEY_CANDIDATES in utils.py once you've
inspected the real file (e.g. `python -c "import scanpy as sc; a =
sc.read_h5ad('...', backed='r'); print(a.obs.columns.tolist(),
list(a.obsm.keys()))"`).

https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html
"""

import scanpy as sc

from utils import (
    CELLTYPE_KEY,
    H5AD_MULTIVI,
    H5AD_MULTIVI_RAW,
    LEIDEN_KEY,
    LEIDEN_RESOLUTION,
    MULTIVI_LATENT_KEY_CANDIDATES,
    MULTIVI_MODEL_DIR,
    MULTIVI_SOURCE_H5AD,
    MUSC_LABEL,
    N_NEIGHBORS,
    SEED,
    configure_plotting,
)


def load_musc_subset():
    """Loads Mingke's MultiVI object and subsets to MuSCs, caching the subset
    to H5AD_MULTIVI_RAW so this step doesn't need to re-run every time."""

    if H5AD_MULTIVI_RAW.exists():
        return sc.read_h5ad(H5AD_MULTIVI_RAW)

    if not MULTIVI_SOURCE_H5AD.exists():
        raise FileNotFoundError(
            f"{MULTIVI_SOURCE_H5AD} not found. Download Mingke's MultiVI object "
            "from SharePoint onto the E3 cluster's storage and either place it "
            "at that path or set the MULTIVI_SOURCE_H5AD env var to its path."
        )

    adata = sc.read_h5ad(MULTIVI_SOURCE_H5AD)

    if CELLTYPE_KEY in adata.obs:
        keep = adata.obs[CELLTYPE_KEY].astype(str) == MUSC_LABEL
        adata = adata[keep].copy()
        print(f"Subsetted to {adata.n_obs} MuSCs using obs['{CELLTYPE_KEY}'] == '{MUSC_LABEL}'")
    else:
        print(
            f"'{CELLTYPE_KEY}' not found in obs -- assuming Mingke's object is "
            "already MuSC-only. Verify this before trusting downstream results."
        )

    if "counts" not in adata.layers:
        print(
            "WARNING: 'counts' layer not found. age_characterization.py's "
            "try_load_mrvi_model() needs raw RNA counts in adata.layers['counts'] "
            "to train its age-effect model -- check whether Mingke's object stores "
            "raw counts under a different layer/in .X and copy it into 'counts' "
            "before running the age analysis on this object."
        )

    adata.write_h5ad(H5AD_MULTIVI_RAW)
    return adata


def get_multivi_latent(adata):
    """Finds a precomputed MultiVI latent embedding in obsm, or trains one."""

    for key in MULTIVI_LATENT_KEY_CANDIDATES:
        if key in adata.obsm:
            print(f"Using precomputed MultiVI latent obsm['{key}']")
            return adata.obsm[key]

    raise KeyError(
        f"None of {MULTIVI_LATENT_KEY_CANDIDATES} found in adata.obsm "
        f"(available: {list(adata.obsm.keys())}). Either Mingke's object stores "
        "the latent under a different key -- add it to MULTIVI_LATENT_KEY_CANDIDATES "
        "in utils.py -- or the object needs MultiVI trained on it here (not yet "
        "implemented: requires separate RNA/ATAC layers set up per "
        "scvi.model.MULTIVI.setup_anndata, which depends on Mingke's exact "
        "modality layout)."
    )


def main():
    configure_plotting()
    sc.settings.n_jobs = 4

    adata = load_musc_subset()
    adata.obsm["X_multivi_latent"] = get_multivi_latent(adata)

    sc.pp.neighbors(adata, use_rep="X_multivi_latent", n_neighbors=N_NEIGHBORS,
                     random_state=SEED)
    sc.tl.umap(adata, random_state=SEED)

    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY,
                 flavor="igraph", n_iterations=2, directed=False,
                 random_state=SEED)
    adata.obs[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].astype(str)

    adata.write_h5ad(H5AD_MULTIVI)
    print(f"Wrote {adata.n_obs} cells x {adata.n_vars} genes -> {H5AD_MULTIVI}")


if __name__ == "__main__":
    main()
