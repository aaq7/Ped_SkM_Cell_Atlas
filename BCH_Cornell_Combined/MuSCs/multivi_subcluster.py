"""
Subclusters the satellite cells using Ben/Mingke's "Cornell_BCH_orig_ident"
MultiVI (RNA+ATAC) bundle.

Mirrors mrvi_subcluster.py's neighbors/UMAP/leiden steps, but starts from a
MultiVI-integrated object instead of training MrVI ourselves. Run this with
PIPELINE_SOURCE=multivi so every downstream script (age_characterization.py,
cornell_comparison.py, pseudotimekernel_trajectory.py, mrvi_plots.py) reads
the MultiVI outputs instead of the MrVI ones -- see utils.py's H5AD_ACTIVE.

Confirmed from the bundle's README.md/training_config.json/package_validation.json
(2026-09-16): despite living in a SharePoint folder full of Ben's separate FAP
work, this specific package (multivi_results.h5ad) is a whole-atlas model --
285,967 cells across all 60 BCH+Cornell samples, batch_key="orig.ident", joint
latent saved at obsm["X_multivi"] -- not FAP-restricted. Two things follow from
that and from package_multivi_model.py's construction of the file:

1. adata.X holds raw RNA counts directly (never normalized) -- there is no
   "counts" layer, so we create one ourselves below, matching what
   mrvi_subcluster.py/age_characterization.py expect.
2. obs was built fresh from Ben's own combined_cell_metadata.tsv.gz, which
   (per the README) carries orig.ident/cohort/has_rna/has_atac -- NOT this
   repo's `celltype` annotation. So MuSC-subsetting can't just filter
   adata.obs[CELLTYPE_KEY] here; we join it in from H5AD_FULL_ATLAS by
   obs_names (barcode) instead. If barcodes don't overlap, that join fails
   loudly rather than silently keeping the whole atlas.

Unverified: var_names are labeled "gene_id" in the packaging script -- confirm
these are HGNC symbols (PAX7, MYF5, ...) and not Ensembl IDs before trusting
CORNELL_FIGURE_GENES/MYO_MARKERS lookups; a mismatch would silently return an
empty gene list rather than error. Also per training_config.json's
design_rationale, Ben built this as a "sample-conditioned sensitivity model
for visualization and cell-state structure", explicitly NOT for donor-level
age/sex inference -- consistent with the AGE_MODEL_DIR caveat in utils.py.

https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html
"""

import scanpy as sc

from utils import (
    CELLTYPE_KEY,
    H5AD_FULL_ATLAS,
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


def attach_celltype(adata):
    """Joins this repo's `celltype` annotation onto Ben's MultiVI object by
    barcode, since his object's obs doesn't carry it (see module docstring)."""

    full_atlas = sc.read_h5ad(H5AD_FULL_ATLAS, backed="r")
    celltype = full_atlas.obs[CELLTYPE_KEY].astype(str)

    overlap = adata.obs_names.isin(celltype.index)
    if overlap.mean() < 0.99:
        raise ValueError(
            f"Only {overlap.mean():.1%} of the MultiVI object's obs_names match "
            f"{H5AD_FULL_ATLAS}'s barcodes -- barcode formats likely differ "
            "between the two objects (e.g. a suffix/prefix mismatch). Inspect "
            "adata.obs_names vs. full_atlas.obs_names before trusting any "
            "MuSC subset built from this join."
        )

    adata.obs[CELLTYPE_KEY] = celltype.reindex(adata.obs_names).values
    return adata


def load_musc_subset():
    """Loads Ben/Mingke's MultiVI object and subsets to MuSCs, caching the
    subset to H5AD_MULTIVI_RAW so this step doesn't need to re-run every time."""

    if H5AD_MULTIVI_RAW.exists():
        return sc.read_h5ad(H5AD_MULTIVI_RAW)

    if not MULTIVI_SOURCE_H5AD.exists():
        raise FileNotFoundError(
            f"{MULTIVI_SOURCE_H5AD} not found. Download multivi_results.h5ad "
            "from the Cornell_BCH_orig_ident SharePoint bundle onto the E3 "
            "cluster's storage and either place it at that path or set the "
            "MULTIVI_SOURCE_H5AD env var to its path."
        )

    adata = sc.read_h5ad(MULTIVI_SOURCE_H5AD)

    # adata.X is raw RNA counts already (package_multivi_model.py never
    # normalizes it) -- mirror musc_subset.py by keeping a "counts" layer.
    adata.layers["counts"] = adata.X.copy()

    if CELLTYPE_KEY not in adata.obs:
        print(f"'{CELLTYPE_KEY}' not in obs -- joining it in from {H5AD_FULL_ATLAS}")
        adata = attach_celltype(adata)

    keep = adata.obs[CELLTYPE_KEY].astype(str) == MUSC_LABEL
    adata = adata[keep].copy()
    print(f"Subsetted to {adata.n_obs} MuSCs using obs['{CELLTYPE_KEY}'] == '{MUSC_LABEL}'")

    looks_like_ensembl = adata.var_names.str.match(r"^ENSG\d+").mean() > 0.5
    if looks_like_ensembl:
        print(
            "WARNING: var_names look like Ensembl gene IDs, not HGNC symbols "
            "(e.g. PAX7, MYF5). MYO_MARKERS/CORNELL_FIGURE_GENES lookups via "
            "present_genes() will silently return an empty list against this "
            "object until var_names are mapped to symbols."
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
