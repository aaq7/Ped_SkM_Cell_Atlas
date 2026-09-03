"""
Subclusters the satellite cells using MrVI.

https://docs.scvi-tools.org/en/1.2.2/tutorials/notebooks/scrna/MrVI_tutorial.html
"""

import pandas as pd
import scanpy as sc
import scipy.io
from scipy import sparse

from utils import (
    EXPORT_DIR,
    H5AD_MRVI,
    H5AD_RAW,
    LEIDEN_KEY,
    LEIDEN_RESOLUTION,
    MRVI_BATCH_SIZE,
    MRVI_MAX_EPOCHS,
    MRVI_MODEL_DIR,
    MRVI_N_LATENT,
    N_NEIGHBORS,
    SAMPLE_KEY,
    SEED,
    configure_plotting,
)


def build_anndata():
    """Constructs the AnnData object from Seurat components"""

    mtx = scipy.io.mmread(EXPORT_DIR / "matrix.mtx")
    genes = pd.read_csv(EXPORT_DIR / "genes.tsv", header=None)[0].astype(str).values
    barcodes = pd.read_csv(EXPORT_DIR / "barcodes.tsv", header=None)[0].astype(str).values
    meta = pd.read_csv(EXPORT_DIR / "metadata.csv")

    # AnnData wants cells x genes --> transpose
    X = sparse.csr_matrix(mtx.T)

    # Ensure that the metadata rows are in the same order as the barcodes in the matrix
    meta = meta.set_index("cell_barcode").loc[barcodes]

    # Instantiate annData object with the transposed matrix, metadata, and gene names
    adata = sc.AnnData(X=X, obs=meta, var=pd.DataFrame(index=genes))
    adata.var_names_make_unique()  

    # Save raw copy of counts for MrVI
    adata.layers["counts"] = adata.X.copy()
    # Write file to h5ad format
    adata.write_h5ad(H5AD_RAW)
    return adata


def preprocess(adata):
    """Filters low-detection genes and log-normalizes."""

    # Remove genes that are detected in fewer than 5 cells
    sc.pp.filter_genes(adata, min_cells=5)

    # Create copy of UMI counts into lognorm layer
    adata.layers["lognorm"] = adata.layers["counts"].copy()
    # Point .X at the copied lognorm layer to preserve original 
    adata.X = adata.layers["lognorm"]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Store normalized log-transformed data in the "lognorm" layer
    adata.layers["lognorm"] = adata.X.copy()
    return adata


def train_mrvi(adata, sample_key):
    """Trains MrVI on raw counts."""

    from scvi.external import MRVI 

    # Skip retraining on reruns 
    if MRVI_MODEL_DIR.exists():
        return MRVI.load(str(MRVI_MODEL_DIR), adata=adata)

    # MrVI models the raw count distribution directly
    MRVI.setup_anndata(adata, layer="counts", sample_key=sample_key)

    model = MRVI(adata, n_latent=MRVI_N_LATENT) # latent dimensionality, recommended 10-30
    model.train(
        max_epochs=MRVI_MAX_EPOCHS, # change this in utils
        batch_size=MRVI_BATCH_SIZE, # 256 cells per batch (optimize memory)
        early_stopping=True, # stop training when performance plataeus   
    )

    MRVI_MODEL_DIR.mkdir(parents=True, exist_ok=True) # make directory for future use
    model.save(str(MRVI_MODEL_DIR), overwrite=True)
    return model


def main():
    configure_plotting()
    sc.settings.n_jobs = 4  # 4 cpu cores 

    # Construct annData object
    adata = build_anndata()
    # Preprocess for MrVI formatting
    adata = preprocess(adata)

    # Train MrVI model on the preprocessed data
    model = train_mrvi(adata, SAMPLE_KEY)

    # Extracts sample-corrected latent space from mrvi model and stores in annData object
    adata.obsm["X_mrvi_u"] = model.get_latent_representation(give_z=False)

    # Construct k-nearest neighbor graph on n = 15 neighbors
    sc.pp.neighbors(adata, use_rep="X_mrvi_u", n_neighbors=N_NEIGHBORS,
                    random_state=SEED)
    # Calculate UMAP 2D embedding
    sc.tl.umap(adata, random_state=SEED)

    # Run leiden clustering graph algoroithm 
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, key_added=LEIDEN_KEY,
                 flavor="igraph", n_iterations=2, directed=False,
                 random_state=SEED)
    adata.obs[LEIDEN_KEY] = adata.obs[LEIDEN_KEY].astype(str)  # avoid category dtype issues on read-back

    # Write formatted annData object to h5ad
    adata.write_h5ad(H5AD_MRVI)


if __name__ == "__main__":
    main()
