import anndata as ad
import numpy as np
from sklearn.preprocessing import LabelEncoder


pathh = f"../synthetic_data_seed_123"

datasetnames = [
    "Adam",
    "Bach",
    "Campbell",
    "Cao_2020_Spleen",
    "Muraro",
    "Quake_10x_Bladder",
    "Quake_10x_Limb_Muscle",
    "Quake_10x_Spleen",
    "Quake_Smart-seq2_Diaphragm",
    "Quake_Smart-seq2_Limb_Muscle",
    "Quake_Smart-seq2_Lung",
    "Quake_Smart-seq2_Trachea",
    "Romanov",
    "Shekhar",
    "Tosches_turtle",
    "Wang_Large_Intestine",
    "Young"]
seeds = [111]


def load_h5ad_data(dataPath):
    adata = ad.read_h5ad(dataPath)

    # X
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # y
    if "cell_type1" in adata.obs.columns:
        y = adata.obs["cell_type1"].astype(str).values
    elif "celltype" in adata.obs.columns:
        y = adata.obs["celltype"].astype(str).values
    else:
        raise ValueError("No cell-type label found in adata.obs")

    # Convert labels to 0, 1, 2, ...
    y = LabelEncoder().fit_transform(y).astype(np.int64)

    return X, y