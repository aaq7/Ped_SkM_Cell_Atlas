"""
Shared paths, parameters, and helpers.
"""

import re
from pathlib import Path

import matplotlib
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
H5AD_RAW = PROJECT_DIR / "musc_raw.h5ad"            # raw unprocessed data
H5AD_MRVI = PROJECT_DIR / "musc_mrvi.h5ad"          # mrvi constructed object
H5AD_CELLRANK = PROJECT_DIR / "musc_cellrank.h5ad"  # cellrank constructed object
MRVI_MODEL_DIR = PROJECT_DIR / "mrvi_model"         # trained MrVI model
MODEL_DIR = MRVI_MODEL_DIR                          # alias, matches MingKe's naming
# MrVI refit on H5AD_MRVI's post-filter gene set (25,739 genes) so it can be reloaded directly
# against that object. MRVI_MODEL_DIR above was trained on the pre-filter gene set (27,442 genes,
# fit before sc.pp.filter_genes() ran in mrvi_subcluster.preprocess), so MRVI.load() against the
# saved h5ad fails on a var-count mismatch -- see age_characterization.py's try_load_mrvi_model.
AGE_MODEL_DIR = PROJECT_DIR / "mrvi_model_age"
FIG_DIR = PROJECT_DIR / "figures"                   # output figures
TABLES_DIR = PROJECT_DIR / "tables"                 # output tables (DE genes, etc.)
RESULT_DIR = PROJECT_DIR / "results_age"            # age characterization output (figures + tables)

SAMPLE_KEY = "orig.ident"   # sample identifier column
DONOR_KEY = "donor_id"      # donor identifier column, derived from SAMPLE_KEY
LEIDEN_KEY = "leiden_musc"  # leiden clustering key in adata.obs
ANNOTATION_KEY = "cluster_label"  # "cluster N" labels derived from LEIDEN_KEY (see annotate_res03)
AGE_KEY = "age_years"       # donor age in years, from load_donor_metadata
AGE_GROUP_KEY = "age_group" # binned age, from AGE_GROUP_BINS/LABELS below

# leiden_musc at LEIDEN_RESOLUTION=0.3 currently yields these 8 clusters (musc_mrvi.h5ad) --
# no manually curated subtype names exist yet, so we use plain cluster labels, consistent
# with cornell_comparison.py. Update this (and re-run) if the clustering changes.
SUBTYPE_ORDER = [f"cluster {i}" for i in range(8)]

# Pediatric age-group bins (years), right-inclusive: (0,1], (1,5], (5,11], (11,100]
AGE_GROUP_BINS = [-0.001, 1, 5, 11, 100]
AGE_GROUP_LABELS = ["Infant", "Early Childhood", "Late Childhood", "Adolescence"]

# Myo markers + targets
MYO_MARKERS = ["PAX7", "PAX3", "MYF5", "MYOD1", "MYOG"]
TARGET_GENES = ["SELENON", "ELL2"]

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

# Clustering
N_NEIGHBORS = 15        # recommended 10-20
LEIDEN_RESOLUTION = 0.3 # adjust accordingly

# CellRank - 80/20 directional to connectivity is default
KERNEL_WEIGHT_DIRECTIONAL = 0.8
KERNEL_WEIGHT_CONNECTIVITY = 0.2


# SAME AS MINGKE (configure_plotting is copied verbatim from his utils.py).
def configure_plotting():
    """Sets figure defaults and creates output folders. From MingKe"""

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    sc.settings.set_figure_params(dpi=120, dpi_save=300, frameon=False, fontsize=9)


# DIFFERENT FROM MINGKE: his savefig() takes a path that's already absolute (built from his own
# RESULT_DIR by the caller) and just does path.parent.mkdir(...) + fig.savefig(...). Several of our
# scripts (mrvi_plots.py, cornell_comparison.py, cellrank_trajectory.py) call savefig() with a bare
# filename like "01_umap_subclusters.png" instead of a full path, so we resolve non-absolute paths
# against FIG_DIR before saving -- otherwise those calls would write into the current working directory.
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


# NOT IN MINGKE'S utils.py: his scripts only ever saved figures, never tables, so he had no CSV
# helper. We add one (mirroring savefig's resolve-against-a-base-dir pattern) since several of our
# analyses (e.g. DE gene tables) need to write csvs to TABLES_DIR.
def savecsv(df, path):
    """Saves a dataframe to tables/."""
    path = Path(path)
    if not path.is_absolute():
        path = TABLES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# SAME AS MINGKE (present_genes is copied verbatim from his utils.py).
def present_genes(adata, genes):
    """Drops genes not in the object. From MingKe"""

    var_names = pd.Index(adata.var_names.astype(str))
    return [gene for gene in genes if gene in var_names]


# NOT IN MINGKE'S utils.py: his H5AD_PATH already had an "age" column on adata.obs, so his
# read_adata()/ensure_age_columns() could compute AGE_KEY directly from adata.obs. Our raw object has
# no per-cell age column at all -- age lives donor-side in a separate metadata CSV (DONOR_META_CSV)
# keyed by donor ID and must be parsed out of free-text strings like "4 yo" / "5 wk" and joined onto
# obs by donor ID (see sample_to_donor_id/load_donor_metadata below, used in mrvi_subcluster.py's
# preprocessing before this h5ad is written). None of this exists in his script.
def sample_to_donor_id(sample_ids):
    """Strips the trailing batch/assay suffix off orig.ident (e.g. 'CTRL100_B' -> 'CTRL100')."""

    return pd.Series(sample_ids).astype(str).str.rsplit("_", n=1).str[0]


# NOT IN MINGKE'S utils.py -- see sample_to_donor_id above.
def parse_age_years(age_str):
    """Parses an age string like '4 yo', '16.0 yo', or '5 wk' into a float number of years."""

    match = re.match(r"([\d.]+)\s*(yo|wk)", str(age_str).strip(), re.IGNORECASE)
    if not match:
        return float("nan")
    value, unit = float(match.group(1)), match.group(2).lower()
    return value / 52 if unit == "wk" else value


# NOT IN MINGKE'S utils.py -- see sample_to_donor_id above. This is our equivalent of the age lookup
# his ensure_age_columns() got for free from adata.obs["age"].
def load_donor_metadata():
    """Loads donor-level metadata (age, sex, tissue, etc.) keyed by donor ID.

    Source csv has one row per donor per assay, so donors profiled with multiple
    assays (e.g. snRNA-seq and snRNA-seq/snATAC-seq) appear more than once;
    de-dupe on donor ID since age/sex/tissue are consistent across those rows.
    """

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


# DIFFERENT FROM MINGKE: his read_adata() calls patch_null_log1p_base() (works around an h5ad where
# uns/log1p/base was written as null, which scanpy's h5ad reader chokes on) and then
# ensure_age_columns(), which computes AGE_KEY from adata.obs["age"] with pd.to_numeric and bins it
# with pd.cut(bins=[-inf,3,6,13,inf]) into his 4 age groups. We don't need patch_null_log1p_base --
# H5AD_MRVI was written by our own mrvi_subcluster.py pipeline (not passed through the R/Seurat export
# that produced his null uns/log1p/base), so that field is never null here. And AGE_KEY (age_years)
# is already a per-cell column on H5AD_MRVI by the time this runs -- it was joined in from
# load_donor_metadata() during preprocessing -- so there's nothing to compute, only the age-group
# binning is left, and it uses AGE_GROUP_BINS/AGE_GROUP_LABELS (pediatric bins: infant/early
# childhood/late childhood/adolescence) instead of his adult-oriented bins, since this cohort is
# entirely pediatric.
def read_adata():
    """Loads the MrVI-subclustered object and attaches the binned age group."""

    adata = sc.read_h5ad(H5AD_MRVI)
    adata.obs[AGE_GROUP_KEY] = pd.Categorical(
        pd.cut(adata.obs[AGE_KEY], bins=AGE_GROUP_BINS, labels=AGE_GROUP_LABELS),
        categories=AGE_GROUP_LABELS,
    )
    return adata


# DIFFERENT FROM MINGKE: his annotate_res03() calls ensure_leiden_res03() first, which computes the
# leiden clustering itself (sc.tl.leiden(resolution=0.3, ...)) if it isn't already in adata.obs, via
# his ensure_neighbors_umap() helper for the neighbors graph/UMAP. Our object always already has
# LEIDEN_KEY/UMAP computed by mrvi_subcluster.py before this ever runs, so there's no
# ensure_leiden_res03/ensure_neighbors_umap equivalent here -- we raise instead of silently
# recomputing, since a missing key means the upstream pipeline step was skipped, not that we should
# quietly redo it with different parameters. His RES03_CLUSTER_TO_SUBTYPE then maps each cluster
# number to one of 6 manually curated FAP subtype names (fillna "Unassigned" for stragglers). No such
# manual annotation exists yet for this dataset's 8 clusters, so we label them "cluster N" (matching
# the plain-cluster-label convention already used in cornell_comparison.py) and raise if the observed
# clusters don't match SUBTYPE_ORDER, instead of mapping through a dict that would silently mislabel
# clusters if the clustering drifted.
def annotate_res03(adata):
    """Labels each cell with its resolution-0.3 Leiden cluster as 'cluster N' (matches cornell_comparison.py).

    No manually curated subtype names exist for this dataset yet, so SUBTYPE_ORDER
    is just the plain cluster labels. Raises if the clustering has drifted from
    what SUBTYPE_ORDER expects, instead of silently mislabeling/dropping clusters.
    """

    if LEIDEN_KEY not in adata.obs:
        raise ValueError(f"'{LEIDEN_KEY}' not found in adata.obs; run mrvi_subcluster.py first")

    clusters = sorted(adata.obs[LEIDEN_KEY].astype(str).unique(), key=int)
    labels = [f"cluster {c}" for c in clusters]
    if labels != SUBTYPE_ORDER:
        raise ValueError(
            f"Leiden clusters {labels} don't match SUBTYPE_ORDER {SUBTYPE_ORDER} in utils.py -- "
            "update SUBTYPE_ORDER if the clustering has changed."
        )

    adata.obs[ANNOTATION_KEY] = pd.Categorical(
        "cluster " + adata.obs[LEIDEN_KEY].astype(str), categories=SUBTYPE_ORDER
    )
    return adata


# DIFFERENT FROM MINGKE: his get_sample_key() searches adata.obs for the first match in a list of
# common sample-id column names ("patient", "sample", "donor_id", "orig.ident", "batch", ...), falling
# back to a synthetic per-cell "_sample_id" if none exist -- because his dataset's sample column name
# wasn't fixed/known in advance. Our dataset always uses Seurat's "orig.ident" (SAMPLE_KEY) as the
# sample column, and each sample is exactly one donor, so there's nothing to search for; this just
# returns the constant. `adata` is accepted (and unused) only to keep the call signature compatible
# with his version.
def get_sample_key(adata=None):
    """Returns the sample identifier column (one sample per donor in this dataset)."""

    return SAMPLE_KEY


# SAME AS MINGKE (same Benjamini-Hochberg algorithm as his bh_fdr -- argsort, cumulative-min from the
# largest p-value down, clip to [0,1] -- just with our own variable names and an isnan mask instead of
# his isfinite mask, which behaves identically since p-values here are never +/-inf).
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR correction. NaN p-values pass through as NaN."""

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


# DIFFERENT FROM MINGKE: his expression_matrix(adata, genes, layer=None) filters genes through
# present_genes() itself, returns an empty DataFrame if none are present, and reads adata.X unless a
# specific `layer` is requested. We don't re-filter or accept a `layer` param because every call site
# in this codebase already calls present_genes() first and this object's "lognorm" layer is exactly
# equal to adata.X (verified: both hold the same log-normalized values) -- no caller here ever needs
# a different layer, so we just read "lognorm" directly when present.
def expression_matrix(adata, genes):
    """Returns a cells x genes DataFrame of (log-normalized, if available) expression."""

    view = adata[:, genes]
    X = view.layers["lognorm"] if "lognorm" in view.layers else view.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pd.DataFrame(np.asarray(X), index=adata.obs_names, columns=genes)


# DIFFERENT FROM MINGKE (fixed to match his intent): his plot_heatmap(matrix, path, cmap="RdBu_r",
# center=0, vmin=None, vmax=None, cbar_label="") declares a `center` parameter but never actually uses
# it in the function body -- when vmin/vmax aren't given, he always auto-scales symmetrically around 0
# via the 95th percentile of |values| (robust to outlier cells/genes), regardless of `center`. That's
# a latent bug in his script (a diverging colormap with a non-zero center would render off-center).
# We keep his percentile auto-scaling as the fallback (same as his) but actually implement `center` --
# several of our calls pass center=0 expecting a symmetric colorbar. We also copied his exact figsize
# formula and his no-ticks/no-spines styling (both absent from our earlier version).
def plot_heatmap(df, path, cmap="viridis", vmin=None, vmax=None, center=None, cbar_label=None):
    """Generic matrix heatmap (rows x cols), saved via savefig."""

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


# DIFFERENT FROM MINGKE: his sample_subtype_fractions(adata, subtype_key=ANNOTATION_KEY) takes the
# subtype column as a parameter (he had multiple candidate annotation columns to choose from); we
# only ever have one (ANNOTATION_KEY), so it's hardcoded. He also drops rows missing sample/subtype/age
# *before* counting, and merges in per-sample age via a separate sample_metadata() helper (median age,
# first non-null age group) that lives in his utils.py; ours computes that inline with drop_duplicates
# since every sample here is exactly one donor with one age value (no medians needed), and the
# equivalent helper (sample_metadata_from_obs) lives in age_characterization.py instead of utils.py
# only because it's needed there too and importing utils.py -> age_characterization.py would be
# circular. Otherwise the reindex-to-zero-fill-missing-subtypes logic and the added "total_cells"
# column are the same as his.
def sample_subtype_fractions(adata):
    """Per-sample subtype composition, with each sample's age/age-group attached."""

    sample_key = get_sample_key(adata)
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


# DIFFERENT FROM MINGKE: his spearman_table(df, value_col, group_cols, age_col=AGE_KEY) takes the age
# column as a parameter (defaulting to his AGE_KEY) so callers could correlate against something else;
# nothing in this project ever does that, so it's hardcoded to AGE_KEY. He also requires >=2 distinct
# values in value_col (only age needs >=3), whereas we require >=3 for both -- age is nearly continuous
# here (mostly-unique ages in years, unlike some of his coarser value columns), so requiring 3 distinct
# fraction/expression values too guards against a degenerate 2-point "correlation" on the value side as
# well.
def spearman_table(df, value_col, group_cols):
    """Per-group Spearman correlation of value_col against AGE_KEY, BH-corrected across groups."""

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


# DIFFERENT FROM MINGKE: his subtype_palette() is a fixed dict of hex colors hand-picked per named FAP
# subtype ("MME+": "#4C78A8", etc.) plus "Unassigned": "#999999", since he had 6 manually curated
# subtype names. We have no curated names yet (SUBTYPE_ORDER is "cluster 0".."cluster 7", see
# annotate_res03), so there are no fixed identities to hand-pick colors for -- we generate a color per
# cluster from matplotlib's tab10 colormap instead, keyed the same way (a dict from SUBTYPE_ORDER entry
# to color) so every other function that consumes this palette works unchanged.
def subtype_palette():
    """Consistent color per subtype/cluster label, in SUBTYPE_ORDER."""

    cmap = plt.get_cmap("tab10")
    return {subtype: cmap(i % 10) for i, subtype in enumerate(SUBTYPE_ORDER)}


# NOT IN MINGKE'S utils.py: his AGE_GROUP_KEY had no dedicated palette/plotting helper -- his figures
# colored panels by subtype (via his subtype_palette()) rather than by age group. Our
# plot_number_age_groups/plot_number_continuous_age color by age group instead (see
# age_characterization.py's module docstring for why), so this palette is new.
# Fixed color per age group, in AGE_GROUP_LABELS order (blue/green/orange/purple)
AGE_GROUP_COLORS = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]


def age_group_palette():
    """Consistent color per age group, in AGE_GROUP_LABELS order."""

    return dict(zip(AGE_GROUP_LABELS, AGE_GROUP_COLORS))
