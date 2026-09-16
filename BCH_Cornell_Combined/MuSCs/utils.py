"""
Shared paths, parameters, and helpers.
"""

import os
import re
from pathlib import Path

import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

# pandas 3 string dtype breaks h5ad writing without this
anndata.settings.allow_write_nullable_strings = True

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = PROJECT_DIR / "export_musc"            # from seurat_export.Rmd
DONOR_META_CSV = EXPORT_DIR / "Ped_SkM_Sample_Meta(Sample Metadata).csv"  # donor-level age/sex/tissue metadata, from Monkol
H5AD_FULL_ATLAS = PROJECT_DIR / "full_atlas_SoupX_MRVI_final.h5ad"  # combined BCH+Cornell atlas, from seurat_export.Rmd
H5AD_RAW = PROJECT_DIR / "musc_raw.h5ad"            # raw unprocessed data
H5AD_MRVI = PROJECT_DIR / "musc_mrvi.h5ad"          # mrvi constructed object
H5AD_PSEUDOTIMEKERNEL = PROJECT_DIR / "musc_pseudotimekernel.h5ad"    # PseudotimeKernel (DPT) output
H5AD_CYTOTRACEKERNEL = PROJECT_DIR / "musc_cytotracekernel.h5ad"      # CytoTRACEKernel output
MRVI_MODEL_DIR = PROJECT_DIR / "mrvi_model"         # trained MrVI model

# --- MultiVI variant -------------------------------------------------------
# Mingke's MultiVI object (RNA+ATAC integration) is a separate starting point
# from the MrVI pipeline above. Set PIPELINE_SOURCE=multivi in the environment
# (e.g. on the E3 cluster) to point every downstream script at the MultiVI
# object/outputs instead, without touching the existing MrVI object or results.
PIPELINE_SOURCE = os.environ.get("PIPELINE_SOURCE", "mrvi").strip().lower()
if PIPELINE_SOURCE not in ("mrvi", "multivi"):
    raise ValueError(f"PIPELINE_SOURCE must be 'mrvi' or 'multivi', got {PIPELINE_SOURCE!r}")

# Mingke's exported MultiVI object -- place the file downloaded from SharePoint
# here (or override via the MULTIVI_SOURCE_H5AD env var) before running
# multivi_subcluster.py. Schema (obs columns, obsm latent key) is not yet
# verified against MRVI's -- see multivi_subcluster.py's checks.
MULTIVI_SOURCE_H5AD = Path(
    os.environ.get("MULTIVI_SOURCE_H5AD", PROJECT_DIR / "full_atlas_MultiVI.h5ad")
)
H5AD_MULTIVI_RAW = PROJECT_DIR / "musc_multivi_raw.h5ad"   # MuSC-subset of Mingke's object
H5AD_MULTIVI = PROJECT_DIR / "musc_multivi.h5ad"           # + neighbors/umap/leiden
MULTIVI_MODEL_DIR = PROJECT_DIR / "multivi_model"          # trained MultiVI model (if we train rather than reuse Mingke's)

# Candidate obsm keys to look for a precomputed MultiVI latent under, in order
# of preference. Confirm the real key once the file is downloaded and update
# this list/utils accordingly -- multivi_subcluster.py fails loudly rather
# than guessing if none of these are present.
MULTIVI_LATENT_KEY_CANDIDATES = ["X_multivi", "X_MultiVI", "MultiVI", "X_multiVI"]

# H5AD_ACTIVE / FIG_DIR / TABLES_DIR / RESULT_DIR below are what every
# downstream script (age_characterization.py, cornell_comparison.py,
# pseudotimekernel_trajectory.py, mrvi_plots.py) actually reads/writes --
# swapping PIPELINE_SOURCE reroutes the whole pipeline in one place, and
# MultiVI outputs land in their own *_multivi dirs so they never clobber the
# existing MrVI figures/tables/results.
H5AD_ACTIVE = H5AD_MULTIVI if PIPELINE_SOURCE == "multivi" else H5AD_MRVI
_SUFFIX = "_multivi" if PIPELINE_SOURCE == "multivi" else ""
FIG_DIR = PROJECT_DIR / f"figures{_SUFFIX}"                 # output figures
TABLES_DIR = PROJECT_DIR / f"tables{_SUFFIX}"               # output tables (DE genes, etc.)
RESULT_DIR = PROJECT_DIR / f"results_age{_SUFFIX}"          # age characterization output (figures + tables)

# IMPORTANT CAVEAT: age_characterization.py's differential_abundance/differential_expression
# age-effect stats (try_load_mrvi_model) are computed by training/loading a *separate*
# scvi.external.MRVI model directly on RNA counts -- scvi-tools has no MultiVI equivalent
# of that test. Running with PIPELINE_SOURCE=multivi changes which cluster/latent groups
# cells for those stats, but the underlying age-effect model itself is still MrVI-on-RNA
# either way, so per Pamela's caution, the numeric age effect sizes may not shift much even
# though the clustering upstream does. AGE_MODEL_DIR is still suffixed below purely so the
# multivi run trains/caches its own model against the multivi-clustered adata's exact
# var_names/cell set, rather than silently loading (or overwriting) the mrvi run's cached
# model, which was fit on a different cell set.
AGE_MODEL_DIR = PROJECT_DIR / f"mrvi_model_age{_SUFFIX}"

SAMPLE_KEY = "orig.ident"   # sample identifier column
DONOR_KEY = "donor_id"      # donor identifier column, derived from SAMPLE_KEY
CELLTYPE_KEY = "celltype"   # full-atlas cell-type annotation column
BATCH_KEY = "batch"         # cohort/assay column (B=BCH, C=Cornell); see mrvi_subcluster.train_mrvi
MUSC_LABEL = "MuSC"         # celltype value identifying satellite cells in the full atlas
LEIDEN_KEY = "leiden_musc"  # leiden clustering key in adata.obs
ANNOTATION_KEY = "cluster_label"  # subtype labels derived from LEIDEN_KEY (see annotate_res03)
AGE_KEY = "age"             # donor age in years, already merged into the atlas upstream (seurat_export.Rmd)
AGE_GROUP_KEY = "age_group" # binned age, from AGE_GROUP_BINS/LABELS below

# leiden_musc at LEIDEN_RESOLUTION=0.3 is best fit; maps leiden cluster id -> subtype name
CLUSTER_NAMES = {
    "0": "Quiescent MuSC 6 (MYF5+)",
    "1": "Quiescent MuSC Unknown1",
    "2": "Quiescent MuSC Unknown2",
    "3": "Quiescent MuSC 4 (Metabolic)",
    "4": "Quiescent MuSC Unknown3",
    "5": "Activated (CXCR4+)/Cycling Progenitor (MKI67+)",
    "6": "Differentiating Myoblast (MYOG+)",
    "7": "Type I (MYH7+)/Type II (MYH1+) committed",
}
SUBTYPE_ORDER = [CLUSTER_NAMES[str(i)] for i in range(8)]

# Pediatric age-group bins (years), right-inclusive: (0,1], (1,5], (5,11], (11,100]
AGE_GROUP_BINS = [-0.001, 1, 5, 11, 100]
AGE_GROUP_LABELS = ["Infant", "Early Childhood", "Late Childhood", "Adolescence"]

# Myo markers
MYO_MARKERS = ["PAX7", "PAX3", "MYF5", "MYOD1", "MYOG"]

# Differential expression (sc.tl.rank_genes_groups, the scanpy equivalent of Seurat's FindAllMarkers)
N_TOP_DE_GENES = 10  # top markers to keep per cluster

# Cornell figure 2A
CORNELL_FIGURE_GENES = [
    "MYF6", "MEF2D", "MYH7B", "MYH1", "MEF2C", "MYMK", "MYMX", "MYOD1", "MYOG",
    "CDKN1C", "CDK4", "CDK1", "MKI67", "CXCR4", "ITGA7", "CEROX1", "MEGF10",
    "MYF5", "MEF2A", "SPRY1", "CALCR", "PAX7", "MEG3",
]

# FROM CORNELL PAPER
CORNELL_MUSC_CLASSES = {
    "Quiescent": ["PAX7", "MEG3", "CALCR", "SPRY1"],
    "Activated/progenitor": ["MEGF10", "CXCR4", "CDK4", "MKI67", "MYF5", "MYOG", "MEF2C"],
    "Myogenic progenitor/committed": ["MYOD1", "MYOG", "MYH7B", "MYH1", "MYH7"],
}

# Candidate gene panels for age-effect analysis: Cornell's MuSC states plus
# classic aging-biology pathways (senescence, inflammation, fibrosis)
CLASSIC_PATHWAYS = {
    **CORNELL_MUSC_CLASSES,
    "Senescence / cell-cycle arrest": ["CDKN1A", "CDKN2A", "CDKN1C", "TP53", "GLB1"],
    "Inflammation / SASP": ["IL6", "IL1B", "CXCL8", "TNF", "NFKB1"],
    "ECM / fibrosis": ["COL1A1", "COL1A2", "COL3A1", "FN1", "TGFB1"],
}

# Purple/black/yellow pallete for heatmap
SEURAT_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "seurat_purple_yellow", ["magenta", "black", "yellow"]
)

# Pink/blue gradient matching the Cornell paper's dotplot (Figure 3B)
CORNELL_DOTPLOT_CMAP = LinearSegmentedColormap.from_list(
    "cornell_pink_blue", ["#e0368f", "#3a53c4"]
)

# MrVI
SEED = 4269
MRVI_MAX_EPOCHS = 100   # number of training cycles the neural network makes
MRVI_N_LATENT = 20      # dimensionality of the latent space
MRVI_BATCH_SIZE = 256   # number of single cells fed into neural network (256 is ideal for memory use)

# MultiVI (only used if multivi_subcluster.py has to train rather than reuse a
# precomputed latent from Mingke's object) -- needs a GPU, unlike MrVI above
MULTIVI_MAX_EPOCHS = 100
MULTIVI_N_LATENT = 20
MULTIVI_BATCH_SIZE = 128

# Clustering
N_NEIGHBORS = 15        # recommended 10-20
LEIDEN_RESOLUTION = 0.3 # adjust accordingly

# CellRank - 80/20 directional to connectivity is default
KERNEL_WEIGHT_DIRECTIONAL = 0.8
KERNEL_WEIGHT_CONNECTIVITY = 0.2

# CellRank GPCCA estimator (CytoTRACEKernel terminal states/fate probabilities/gene trends)
GPCCA_N_COMPONENTS = 20        # number of Schur vectors to compute
GPCCA_N_STATES = 8             # macrostates to fit, matches SUBTYPE_ORDER's 8 leiden_musc clusters
GPCCA_N_TERMINAL_STATES = 3    # macrostates to keep as terminal (method="top_n")


def configure_plotting():
    """Sets figure defaults and creates output folders. From MingKe"""

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    sc.settings.set_figure_params(dpi=120, dpi_save=300, frameon=False, fontsize=9)

def savefig(path, fig=None, close=True):
    """Saves current figure to figures/. From MingKe"""
    path = Path(path)
    if not path.is_absolute():
        path = FIG_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)

def savecsv(df, path):
    """Saves a dataframe to tables/."""
    path = Path(path)
    if not path.is_absolute():
        path = TABLES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def present_genes(adata, genes):
    """Drops genes not in the object. From MingKe"""

    var_names = pd.Index(adata.var_names.astype(str))
    return [gene for gene in genes if gene in var_names]

def sample_to_donor_id(sample_ids):
    """Strips the trailing batch/assay suffix off orig.ident (e.g. 'CTRL100_B' -> 'CTRL100')."""

    return pd.Series(sample_ids).astype(str).str.rsplit("_", n=1).str[0]

def parse_age_years(age_str):
    """Parses an age string like '4 yo', '16.0 yo', or '5 wk' into a float number of years."""

    match = re.match(r"([\d.]+)\s*(yo|wk)", str(age_str).strip(), re.IGNORECASE)
    if not match:
        return float("nan")
    value, unit = float(match.group(1)), match.group(2).lower()
    return value / 52 if unit == "wk" else value

def load_donor_metadata():
    """Loads donor-level metadata (age, sex, tissue, etc.) keyed by donor ID."""

    donor_meta = pd.read_csv(DONOR_META_CSV)
    donor_meta = donor_meta.rename(columns={
        "Donor ID": DONOR_KEY,
        "Age": "age_raw",
        "Sex": "donor_sex",
        "Tissue": "donor_tissue",
        "Source": "donor_source",
        "Reported Ancestry": "reported_ancestry",
    })
    donor_meta["age_years"] = donor_meta["age_raw"].apply(parse_age_years)
    donor_meta = donor_meta.drop_duplicates(subset=DONOR_KEY, keep="first")

    return donor_meta[[
        DONOR_KEY, "age_years", "age_raw", "donor_sex", "donor_tissue",
        "donor_source", "reported_ancestry",
    ]]

def read_adata():
    """Loads the subclustered object (MrVI or MultiVI, per PIPELINE_SOURCE) and
    attaches the binned age group."""

    adata = sc.read_h5ad(H5AD_ACTIVE)
    adata.obs[AGE_GROUP_KEY] = pd.Categorical(
        pd.cut(adata.obs[AGE_KEY], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS),
        categories=AGE_GROUP_LABELS,
    )
    return adata

def annotate_res03(adata):
    """Labels each cell with its resolution-0.3 Leiden cluster's subtype name (see CLUSTER_NAMES)."""

    if LEIDEN_KEY not in adata.obs:
        raise ValueError(f"'{LEIDEN_KEY}' not found in adata.obs; run mrvi_subcluster.py first")

    adata.obs[ANNOTATION_KEY] = pd.Categorical(
        adata.obs[LEIDEN_KEY].astype(str).map(CLUSTER_NAMES), categories=SUBTYPE_ORDER
    )
    return adata

def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. NaN p-values pass through as NaN. From MingKe"""

    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full_like(pvals, np.nan)
    mask = ~np.isnan(pvals)
    n = mask.sum()
    if n == 0:
        return qvals

    p = pvals[mask]
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0, 1)

    out = np.empty(n)
    out[order] = ranked
    qvals[mask] = out
    return qvals

LATENT_OBSM_KEYS = ["X_mrvi_z", "X_mrvi_u", "X_multivi_latent"]  # checked in order, first match wins


def get_cell_latent(adata):
    """Per-cell integrated latent representation, whichever of LATENT_OBSM_KEYS is
    present (MrVI's or MultiVI's) -- falls back to the first 50 columns of .X
    (raw/lognorm expression, NOT a real latent) only if neither is found, with a
    warning, since that fallback silently produces meaningless "latent dims"."""

    for key in LATENT_OBSM_KEYS:
        if key in adata.obsm:
            return np.asarray(adata.obsm[key])

    import warnings
    warnings.warn(
        f"No latent obsm key found (looked for {LATENT_OBSM_KEYS}); falling back "
        "to the first 50 columns of adata.X, which are NOT a real latent space."
    )
    return np.asarray(adata.X[:, : min(50, adata.n_vars)])


def expression_matrix(adata, genes):
    """Returns a cells x genes DataFrame of (log-normalized, if available) expression. From MingKe"""

    view = adata[:, genes]
    X = view.layers["lognorm"] if "lognorm" in view.layers else view.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(np.asarray(X), index=adata.obs_names, columns=genes)

def plot_heatmap(df, path, cmap="viridis", vmin=None, vmax=None, center=None, cbar_label=None):
    """Generic matrix heatmap (rows x cols), saved via savefig. From MingKe"""

    configure_plotting()
    matrix = pd.DataFrame(df)
    data = matrix.to_numpy(dtype=float)

    if center is not None:
        finite = data[np.isfinite(data)]
        max_abs = np.max(np.abs(finite - center)) if finite.size else 1.0
        vmin = center - max_abs if vmin is None else vmin
        vmax = center + max_abs if vmax is None else vmax
    elif vmin is None or vmax is None:
        finite = data[np.isfinite(data)]
        if finite.size:
            lim = np.nanpercentile(np.abs(finite), 95)
            vmin = -lim if vmin is None else vmin
            vmax = lim if vmax is None else vmax

    height = max(2.6, 0.32 * matrix.shape[0] + 1.2)
    width = max(4.5, 0.28 * matrix.shape[1] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label)
    savefig(path, fig)

def sample_subtype_fractions(adata):
    """Per-sample subtype composition, with each sample's age/age-group attached."""

    sample_key = SAMPLE_KEY
    df = (
        adata.obs[[sample_key, ANNOTATION_KEY, AGE_KEY, AGE_GROUP_KEY]]
        .dropna(subset=[sample_key, ANNOTATION_KEY, AGE_KEY])
        .copy()
    )
    counts = df.groupby([sample_key, ANNOTATION_KEY], observed=True).size().rename("n").reset_index()
    totals = counts.groupby(sample_key, observed=True)["n"].transform("sum")
    counts["fraction"] = counts["n"] / totals

    # Fill in (sample, subtype) combos with zero cells as 0, rather than omitting
    # them -- otherwise a sample lacking a subtype entirely would be silently
    # dropped from that subtype's distribution instead of counting as a true 0.
    grid = pd.MultiIndex.from_product(
        [df[sample_key].dropna().unique(), SUBTYPE_ORDER], names=[sample_key, ANNOTATION_KEY]
    )
    counts = counts.set_index([sample_key, ANNOTATION_KEY]).reindex(grid, fill_value=0).reset_index()
    counts["total_cells"] = counts.groupby(sample_key, observed=True)["n"].transform("sum")

    sample_meta = (
        adata.obs[[sample_key, AGE_KEY, AGE_GROUP_KEY]]
        .drop_duplicates(subset=sample_key)
        .set_index(sample_key)
    )
    counts = counts.join(sample_meta, on=sample_key)
    counts = counts.rename(columns={sample_key: "sample", ANNOTATION_KEY: "subtype"})
    counts["sample"] = counts["sample"].astype(str)
    return counts

def spearman_table(df, value_col, group_cols):
    """Per-group Spearman correlation of value_col against AGE_KEY, BH-corrected across groups. From MingKe"""

    clean = df.dropna(subset=[AGE_KEY, value_col])
    rows = []
    for keys, sub in clean.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if sub[AGE_KEY].nunique() < 3 or sub[value_col].nunique() < 3:
            rho, pval = np.nan, np.nan
        else:
            rho, pval = spearmanr(sub[AGE_KEY], sub[value_col])
        rows.append({**dict(zip(group_cols, keys)), "rho": rho, "pval": pval})

    out = pd.DataFrame(rows)
    out["qval"] = bh_fdr(out["pval"].values)
    return out

def subtype_palette():
    """Consistent color per subtype/cluster label, in SUBTYPE_ORDER."""

    cmap = plt.get_cmap("tab10")
    return {subtype: cmap(i % 10) for i, subtype in enumerate(SUBTYPE_ORDER)}


AGE_GROUP_COLORS = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]


def age_group_palette():
    """Consistent color per age group, in AGE_GROUP_LABELS order."""

    return dict(zip(AGE_GROUP_LABELS, AGE_GROUP_COLORS))
