import os
import math
from collections import defaultdict

import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import (accuracy_score,
                             adjusted_rand_score,
                             normalized_mutual_info_score)

# ----------------------------- CONFIG ---------------------------------------
# MODELS      = ["scAce","scAGCL","scCDCG","scCGNet","scDeepCluster","scMAE","scRCL"]
MODELS      = [ "scCGNet", "scDeepCluster", "scMAE", "scRCL"]
FILE_TMPL   = "{model}.csv"          # e.g. "scCGNet.csv"; change ext if needed
READ_KW     = dict()                 # e.g. dict(sep="\t") for TSV
COL_ID      = "cell_id"
COL_ALIGNED = "pred_label_aligned"
COL_TRUE    = "true_label"
dataset_name = "Adam"
MODE           = "leave_one_out"            # "common" (includes your model) or "leave_one_out"
DO_SWEEP       = True                # sweep threshold from majority -> unanimous-correct
SWEEP_MIN_FRAC = 0.50                # lowest fraction to start the sweep at
OUT_DIR        = f"Output/{dataset_name}"                 # where results CSV and plot are written
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------------------------------------------------------
datasetnam = [
        "Adam",
        "Muraro",
        "Quake_10x_Bladder",
        "Quake_10x_Limb_Muscle",
        "Quake_10x_Spleen",
        "Quake_Smart-seq2_Diaphragm",
        "Quake_Smart-seq2_Limb_Muscle",
        "Quake_Smart-seq2_Lung",
        "Quake_Smart-seq2_Trachea",
        "Romanov",
        "Young",
        "Shekhar",
        "Tosches_turtle",
        "Wang_Large_Intestine"
        "Campbell",
        "Cao_2020_Spleen",
        "Bach"
        ]



def load_model(model):
    path = f"cluster_data/{dataset_name}/{FILE_TMPL.format(model=model)}"
    df = pd.read_csv(path, **READ_KW)
    missing = {COL_ID, COL_ALIGNED, COL_TRUE} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = df[[COL_ID, COL_ALIGNED, COL_TRUE]].copy()
    out = out.rename(columns={COL_ALIGNED: f"aligned_{model}",
                              COL_TRUE:    f"true_{model}"})
    return out


def build_table():
    frames = [load_model(m) for m in MODELS]
    sizes = {m: len(f) for m, f in zip(MODELS, frames)}
    merged = reduce(lambda a, b: a.merge(b, on=COL_ID, how="inner"), frames)
    print("Rows per file:", sizes)
    print(f"Rows after inner join on {COL_ID}: {len(merged)}")
    dropped = {m: sizes[m] - len(merged) for m in MODELS}
    if any(v > 0 for v in dropped.values()):
        print("WARNING: join dropped cells (id mismatch across files):", dropped)

    true_cols = [f"true_{m}" for m in MODELS]
    consistent = merged[true_cols].nunique(axis=1).eq(1)
    if not consistent.all():
        n_bad = int((~consistent).sum())
        raise ValueError(f"{n_bad} cells have DIFFERENT true_label across files.")
    merged["true_label"] = merged[true_cols[0]]
    return merged.drop(columns=true_cols)


def evaluate(y_true, y_pred, verbose=True):
    """Compute ACC, ARI, NMI and return them as a dict."""
    acc = accuracy_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    if verbose:
        print(f"ACC = {acc:.4f}")
        print(f"ARI = {ari:.4f}")
        print(f"NMI = {nmi:.4f}")

    return acc, ari, nmi


def mainw(datasetnmaes):
    for dataset in datasetnmaes:
        agrement_cell_acc=defaultdict(list)
        agrement_cell_ari=defaultdict(list)
        agrement_cell_nmi=defaultdict(list)
        agrement_cell= {}
        result=defaultdict(list)
        df = build_table()
        for row in df.iterrows():
            true_labe=row["true_label"]
            tot=0
            for col in df.columns:
                if col not in ["true_label", "cell_id"]:
                    pred_labe=row[col]
                    if int(true_labe)==int(pred_labe):
                        tot+=1
            agrement_cell[row["cell_id"]]=tot

        for i in [10,20,30,40,50,60,70,80,90,100]:
            mask=[]
            for cell in agrement_cell:
                if agrement_cell[cell]>=i:
                    mask.append(True)
                else:
                    mask.append(False)
            result[i] = mask
            true_label = df["true_label"][mask]
            agrement_cell_acc[model_name].append(i)
            for col in df.columns:
                if col not in ["true_label", "cell_id"]:
                   pred_label=df[col][mask]
                   acc,ari,nmi=evaluate(true_label,pred_label)
                   model_name=col.split("_")[1]
                   agrement_cell_acc[model_name].append(acc)




if __name__ == "__main__":
    main()


