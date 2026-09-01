"""
Shared paths, parameters, and helpers.
"""

from pathlib import Path

import matplotlib
import anndata
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

# pandas 3 string dtype breaks h5ad writing without this
anndata.settings.allow_write_nullable_strings = True

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = PROJECT_DIR / "export_musc"            # from seurat_export.Rmd
H5AD_RAW = PROJECT_DIR / "musc_raw.h5ad"            # raw unprocessed data
H5AD_MRVI = PROJECT_DIR / "musc_mrvi.h5ad"          # mrvi constructed object
H5AD_CELLRANK = PROJECT_DIR / "musc_cellrank.h5ad"  # cellrank constructed object
MRVI_MODEL_DIR = PROJECT_DIR / "mrvi_model"         # trained MrVI model
FIG_DIR = PROJECT_DIR / "figures"                   # output figures

SAMPLE_KEY = "orig.ident"   # sample identifier column 
LEIDEN_KEY = "leiden_musc"  # leiden clustering key in adata.obs

# Myo markers + targets
MYO_MARKERS = ["PAX7", "MYF5", "MYOD1", "MYOG"]
TARGET_GENES = ["SELENON", "ELL2"]

# MrVI
SEED = 4269             
MRVI_MAX_EPOCHS = 100   # number of training cycles the neural network makes   
MRVI_N_LATENT = 20      # dimensionality of the latent space
MRVI_BATCH_SIZE = 256   # number of single cells fed into neural network (256 is ideal for memory use)

# Clustering
N_NEIGHBORS = 15        # recommended 10-20
LEIDEN_RESOLUTION = 0.2 # adjust accordingly

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
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(path, bbox_inches="tight")
    if close:
        plt.close(fig)


def present_genes(adata, genes):
    """Drops genes not in the object. From MingKe"""

    var_names = pd.Index(adata.var_names.astype(str))
    return [gene for gene in genes if gene in var_names]
