"""
Align predicted cluster labels to ground-truth labels (Hungarian algorithm)
and save cell barcodes + true labels + aligned predicted labels to a file.

Why this is needed:
  Cluster IDs are arbitrary. Predicted "cluster 3" may correspond to true
  class 1. ARI / NMI don't care about this (they are permutation-invariant),
  but ACC, confusion matrices, and side-by-side visualisation DO. This maps
  each predicted cluster onto the true label it overlaps with most, using the
  optimal one-to-one assignment.

Assumptions:
  - true labels and predicted labels are already over the SAME cells in the
    SAME order (element i in both refers to the same cell). If your two
    pipelines filtered different cells, align on barcodes FIRST — see the
    `align_on_barcodes` helper at the bottom.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import random
import torch
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)



# Fixed column order so appends from different scripts always line up.
RESULTS_COLUMNS = ["model", "dataset", "seed", "ACC","ARI", "NMI"]


def log_result(model_name, dataset_name, metrics,
               results_path="all_results.csv", overwrite_existing=True,seed=42):
    """
    Append one (model, dataset, metrics) row to a shared results file.
    Creates the file on the first call, appends on every later call —
    so separate scripts run at different times all land in one file.

    overwrite_existing:
        True  -> if (model, dataset) already exists, replace that row
                 (re-running a config updates its result).
        False -> always append, even if it duplicates a (model, dataset).
    """
    new_row = pd.DataFrame(
        [{
            "model": model_name,
            "dataset": dataset_name,
            "seed":seed,
            "ACC": metrics["ACC"],
            "ARI": metrics["ARI"],
            "NMI": metrics["NMI"]
        }],
        columns=RESULTS_COLUMNS,
    )

    if os.path.exists(results_path):
        existing = pd.read_csv(results_path)
        if overwrite_existing:
            keep = ~((existing["model"] == model_name) &
                     (existing["dataset"] == dataset_name)&
                     (existing["seed"] == seed))
            existing = existing[keep]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_csv(results_path, index=False)
    return combined


def align_clusters_to_truth(y_true, y_pred):
    """
    Map predicted cluster ids onto true label ids via the Hungarian algorithm
    (maximising total overlap). Handles the case where the number of predicted
    clusters differs from the number of true classes.

    Returns
    -------
    y_pred_aligned : np.ndarray
        Predicted labels re-expressed in the true-label space.
    mapping : dict
        {original_predicted_id: assigned_true_id}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_classes = np.unique(y_true)
    pred_clusters = np.unique(y_pred)

    # Contingency (overlap) matrix: rows = predicted clusters, cols = true classes
    overlap = np.zeros((len(pred_clusters), len(true_classes)), dtype=np.int64)
    pred_index = {c: i for i, c in enumerate(pred_clusters)}
    true_index = {c: j for j, c in enumerate(true_classes)}
    for t, p in zip(y_true, y_pred):
        overlap[pred_index[p], true_index[t]] += 1

    # Hungarian solves a MINIMISATION, so negate to maximise overlap.
    # linear_sum_assignment handles rectangular matrices: it matches
    # min(n_pred, n_true) pairs optimally.
    row_ind, col_ind = linear_sum_assignment(-overlap)

    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[pred_clusters[r]] = true_classes[c]

    # Fallback for any predicted cluster left unmatched (happens when there are
    # more predicted clusters than true classes): assign it to the true class it
    # overlaps with most, so no cell is left without a label.
    for r, cluster in enumerate(pred_clusters):
        if cluster not in mapping:
            best_true = true_classes[np.argmax(overlap[r])]
            mapping[cluster] = best_true

    y_pred_aligned = np.array([mapping[p] for p in y_pred])
    return y_pred_aligned, mapping


def align_on_barcodes(true_df, pred_df, barcode_col, label_col):
    """
    Optional pre-step: if your true labels and predicted labels come from
    pipelines that kept different cells, intersect on barcodes and return
    two label vectors guaranteed to be over the SAME cells in the SAME order.

    true_df / pred_df : DataFrames each containing a barcode column and a
                        label column.
    """
    merged = true_df[[barcode_col, label_col]].merge(
        pred_df[[barcode_col, label_col]],
        on=barcode_col,
        suffixes=("_true", "_pred"),
    )
    barcodes = merged[barcode_col].to_numpy()
    y_true = merged[f"{label_col}_true"].to_numpy()
    y_pred = merged[f"{label_col}_pred"].to_numpy()
    return barcodes, y_true, y_pred


def run(cell_ids, y_true, y_pred, model_name, dataset_name,
        results_path="../all_results.csv", out_path="aligned_labels.csv",seed=42,):
    """
    Align, report metrics, and save cell_id + true + predicted(raw & aligned).
    """
    cell_ids = np.asarray(cell_ids)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if not (len(cell_ids) == len(y_true) == len(y_pred)):
        raise ValueError(
            f"Length mismatch: cells={len(cell_ids)}, "
            f"true={len(y_true)}, pred={len(y_pred)}. "
            "The three arrays must be over the same cells in the same order."
        )

    y_pred_aligned, mapping = align_clusters_to_truth(y_true, y_pred)

    # Metrics. ARI/NMI are computed on the RAW predictions on purpose:
    # they are permutation-invariant, so alignment does not change them.
    # ACC is computed on the ALIGNED predictions, where it is meaningful.
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    acc = float(np.mean(y_true == y_pred_aligned))

    df = pd.DataFrame(
        {
            "cell_id": cell_ids,
            "true_label": y_true,
            "pred_label_raw": y_pred,
            "pred_label_aligned": y_pred_aligned,
        }
    )
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} cells -> {out_path}")
    print(f"ARI (permutation-invariant): {ari:.4f}")
    print(f"NMI (permutation-invariant): {nmi:.4f}")
    print(f"ACC (after Hungarian align):  {acc:.4f}")
    print(f"cluster -> true-class mapping: {mapping}")
    metrics = {"ARI": ari, "NMI": nmi, "ACC": acc}
    print("results saved in:", os.path.abspath("all_results.csv"))
    # log_result(model_name, dataset_name, metrics, results_path=results_path,seed=seed)
    return df, mapping,  metrics


def set_seed(seed=42, deterministic=True):

    random.seed(seed)
    # Hash-based operations (dict/set ordering in some contexts)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU — only if CUDA is actually available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed



def get_my_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Replace this synthetic block with your real data, e.g.:
    #   cell_ids = np.load("barcodes.npy")            # or df["barcode"]
    #   y_true   = np.load("true_labels.npy")
    #   y_pred   = np.load("pred_labels.npy")
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    n = 500
    cell_ids = np.array([f"CELL_{i:04d}" for i in range(n)])
    y_true = rng.integers(0, 4, size=n)
    # predicted: mostly correct but with SHUFFLED ids (2->0, 0->1, ...) + noise
    relabel = {0: 2, 1: 3, 2: 0, 3: 1}
    y_pred = np.array([relabel[t] for t in y_true])
    flip = rng.random(n) < 0.15  # 15% noise
    y_pred[flip] = rng.integers(0, 4, size=flip.sum())

    run(cell_ids, y_true, y_pred, out_path="/home/claude/aligned_labels.csv")