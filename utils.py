"""
Shared paths, parameters, and helpers.
"""

import re
from pathlib import Path

import matplotlib
import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
from matplotlib.colors import LinearSegmentedColormap

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
FIG_DIR = PROJECT_DIR / "figures"                   # output figures
TABLES_DIR = PROJECT_DIR / "tables"                 # output tables (DE genes, etc.)

SAMPLE_KEY = "orig.ident"   # sample identifier column
DONOR_KEY = "donor_id"      # donor identifier column, derived from SAMPLE_KEY
LEIDEN_KEY = "leiden_musc"  # leiden clustering key in adata.obs

# Myo markers + targets
MYO_MARKERS = ["PAX7", "PAX3", "MYF5", "MYOD1", "MYOG"]
TARGET_GENES = ["SELENON", "ELL2"]

# Differential expression (sc.tl.rank_genes_groups, the scanpy equivalent of Seurat's FindAllMarkers)
N_TOP_DE_GENES = 10  # top markers to keep per cluster

# FROM CORNELL PAPER
CORNELL_MUSC_CLASSES = {
    "Quiescent": ["PAX7", "MEG3", "CALCR", "SPRY1"],
    "Activated/progenitor": ["MEGF10", "CXCR4", "CDK4", "MKI67", "MYF5", "MYOG", "MEF2C"],
    "Myogenic progenitor/committed": ["MYOD1", "MYOG", "MYH7B", "MYH1", "MYH7"],
}

# Exact gene list from Cornell's figure 2A 
CORNELL_FIGURE_GENES = [
    "MYF6", "MEF2D", "MYH7B", "MYH1", "MEF2C", "MYMK", "MYMX", "MYOD1", "MYOG",
    "CDKN1C", "CDK4", "CDK1", "MKI67", "CXCR4", "ITGA7", "CEROX1", "MEGF10",
    "MYF5", "MEF2A", "SPRY1", "CALCR", "PAX7", "MEG3",
]

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
