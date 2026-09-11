#!/usr/bin/env python3
"""
Anchor-based model evaluation with accuracy, ARI, and NMI, plus a threshold sweep.

Anchor definition (CORRECTNESS-BASED):
  An "anchor" cell is one that ENOUGH models predict CORRECTLY, i.e. aligned == true_label.
  Correctness is counted across ALL models in MODELS, INCLUDING your own model
  (scCGNet). These are the easy cells. The remaining NON-anchor cells are the hard
  ones, where fewer than the required number of models get the label right.

  IMPORTANT (how this differs from a consensus anchor):
    Here true_label is used to BUILD the split, not only to score it. Consequence:
    a non-anchor cell is by construction a cell that few models get right, so a low
    non-anchor accuracy is partly baked into the selection rather than being pure
    evidence of weakness. Read the SWEEP CURVE comparatively (which model stays
    highest as the correctness bar tightens), not the absolute number at any single
    threshold. Because the scored model is included in the count (common mode),
    cells it gets wrong are pushed into the non-anchor set, which tends to depress
    that model's own non-anchor accuracy. That is the conservative, non-cherry-picked
    choice you asked for.

Anchor rule:
  A cell is an anchor if at least a given fraction of the models predict the correct
  label. The fraction is converted to a required count via ceil, so it behaves
  consistently for any number of models. Example: 0.80 -> >= ceil(0.80*n) models
  correct. 1.0 = every model correct (all agree with the truth).

Sweep:
  No single threshold "proves" a model. A strong model STAYS high as the correctness
  bar is raised toward "all models must be right". So we sweep the required fraction
  from simple majority up to unanimous-correct and report each model's non-anchor
  acc/ari/nmi with the non-anchor n at every level. Read the CURVE.

Metrics (permutation note):
  ari/nmi are permutation-invariant (same on predicted or aligned); only acc uses
  your alignment. All three are computed within the subset.

Modes:
  common        one shared non-anchor set from correctness across ALL models. This is
                the "including my model" setting; scores are directly comparable
                across models (each model helped define the split via its own
                correctness).
  leave_one_out when scoring a model, the anchor is built from whether the OTHER
                models are correct (excludes the scored model). Each model then gets
                a different non-anchor set, so numbers are not strictly comparable
                across models. Use this only if you want the scored model kept out of
                its own split.

Requires: pandas, numpy, scikit-learn, matplotlib (matplotlib optional; skipped if absent)
"""

import os
import math
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.metrics import (accuracy_score,
                             adjusted_rand_score,
                             normalized_mutual_info_score)

# ----------------------------- CONFIG ---------------------------------------
# MODELS      = ["scAce","scAGCL","scCDCG","scCGNet","scDeepCluster","scMAE","scRCL"]
MODELS      = ["scAGCL", "scCGNet", "scDeepCluster", "scMAE", "scRCL"]
FILE_TMPL   = "{model}.csv"          # e.g. "scCGNet.csv"; change ext if needed
READ_KW     = dict()                 # e.g. dict(sep="\t") for TSV
COL_ID      = "cell_id"
COL_ALIGNED = "pred_label_aligned"
COL_TRUE    = "true_label"
dataset_name = "Muraro"
MODE           = "common"            # "common" (includes your model) or "leave_one_out"
DO_SWEEP       = True                # sweep threshold from majority -> unanimous-correct
SWEEP_MIN_FRAC = 0.50                # lowest fraction to start the sweep at
OUT_DIR        = f"Output/{dataset_name}"                 # where results CSV and plot are written
os.makedirs(OUT_DIR, exist_ok=True)
# ----------------------------------------------------------------------------


def threshold_count(frac, n):
    """Smallest integer c such that c/n >= frac ('at least frac%')."""
    c = int(math.ceil(frac * n - 1e-9))
    return max(1, min(c, n))


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


def correct_count(df, models):
    """Per row: how many of `models` predicted the CORRECT (true) label.

    Anchor is defined on this count: a cell with a high correct-count is easy
    (most models get it right); a low correct-count is a hard / non-anchor cell.
    """
    cols = [f"aligned_{m}" for m in models]
    sub = df[cols]
    count = sub.eq(df["true_label"], axis=0).sum(axis=1)
    return count


def _metrics(df, mask, model):
    sub = df[mask]
    if len(sub) == 0:
        return np.nan, np.nan, np.nan
    yt, yp = sub["true_label"], sub[f"aligned_{model}"]
    return (accuracy_score(yt, yp),
            adjusted_rand_score(yt, yp),
            normalized_mutual_info_score(yt, yp))


def metrics_on_subset(df, mask, models, label):
    rows = []
    for m in models:
        acc, ari, nmi = _metrics(df, mask, m)
        rows.append({"model": m, "subset": label, "n": int(mask.sum()),
                     "acc": acc, "ari": ari, "nmi": nmi})
    return pd.DataFrame(rows)


# --------------------------- single-threshold run ---------------------------
def run_single(df):
    if MODE == "common":
        n = len(MODELS)
        thr = threshold_count(SWEEP_MIN_FRAC, n)  # reuse min frac as the single point
        count = correct_count(df, MODELS)
        anchor = count >= thr
        print(f"\n[common] need >= {thr}/{n} models correct  "
              f"anchor={int(anchor.sum())}  non-anchor={int((~anchor).sum())}")
        for lab, mask in [("NON-ANCHOR (hard)", ~anchor),
                          ("ANCHOR (easy)", anchor),
                          ("OVERALL", pd.Series(True, index=df.index))]:
            t = metrics_on_subset(df, mask, MODELS, lab)
            print(f"\n=== {lab} ===")
            print(t[["model", "n", "acc", "ari", "nmi"]].to_string(index=False))
    else:
        rows = []
        for m in MODELS:
            others = [x for x in MODELS if x != m]
            thr = threshold_count(SWEEP_MIN_FRAC, len(others))
            count = correct_count(df, others)
            acc, ari, nmi = _metrics(df, ~(count >= thr), m)
            rows.append({"model": m, "thr": thr, "n": int((~(count >= thr)).sum()),
                         "acc": acc, "ari": ari, "nmi": nmi})
        print(pd.DataFrame(rows).sort_values("acc", ascending=False)
              .to_string(index=False))


# ------------------------------- sweep --------------------------------------
def sweep(df):
    records = []
    if MODE == "common":
        n = len(MODELS)
        lo = threshold_count(SWEEP_MIN_FRAC, n)
        count = correct_count(df, MODELS)
        for thr in range(lo, n + 1):
            non = ~(count >= thr)
            for m in MODELS:
                acc, ari, nmi = _metrics(df, non, m)
                records.append({"thr": thr, "pct": thr / n, "model": m,
                                "n_non_anchor": int(non.sum()),
                                "acc": acc, "ari": ari, "nmi": nmi})
    else:  # leave_one_out
        for m in MODELS:
            others = [x for x in MODELS if x != m]
            n = len(others)
            lo = threshold_count(SWEEP_MIN_FRAC, n)
            count = correct_count(df, others)
            for thr in range(lo, n + 1):
                non = ~(count >= thr)
                acc, ari, nmi = _metrics(df, non, m)
                records.append({"thr": thr, "pct": thr / n, "model": m,
                                "n_non_anchor": int(non.sum()),
                                "acc": acc, "ari": ari, "nmi": nmi})
    res = pd.DataFrame(records)
    res["pct_label"] = res["thr"].astype(str) + "/" + \
        res.apply(lambda r: str(round(r["thr"] / r["pct"])), axis=1) + \
        " (" + (res["pct"] * 100).round(0).astype(int).astype(str) + "%)"

    # Printed pivots
    for metric in ["acc", "ari", "nmi"]:
        piv = res.pivot_table(index="pct_label", columns="model", values=metric)
        print(f"\n=== NON-ANCHOR {metric.upper()} vs threshold (rows = % of models correct) ===")
        print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    npiv = res.pivot_table(index="pct_label", columns="model", values="n_non_anchor")
    print("\n=== non-anchor n at each threshold ===")
    print(npiv.astype("Int64").to_string())

    out_csv = os.path.join(OUT_DIR, "anchor_sweep_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved tidy results -> {out_csv}")

    # Plot (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        xcol = "pct"
        for ax, metric in zip(axes, ["acc", "ari", "nmi"]):
            for m in MODELS:
                d = res[res.model == m].sort_values(xcol)
                ax.plot(d[xcol] * 100, d[metric], marker="o", label=m)
            ax.set_title(f"non-anchor {metric.upper()}")
            ax.set_xlabel("anchor threshold (% of models correct)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("score")
        # annotate non-anchor n (same across models in common mode) on acc panel
        if MODE == "common":
            d0 = res[res.model == MODELS[0]].sort_values(xcol)
            for _, r in d0.iterrows():
                axes[0].annotate(f"n={int(r['n_non_anchor'])}",
                                 (r[xcol] * 100, r["acc"]),
                                 textcoords="offset points", xytext=(0, 6),
                                 fontsize=7, ha="center")
        axes[-1].legend(fontsize=7, loc="best")
        fig.suptitle(f"Non-anchor performance sweep ({MODE}) — a strong model "
                     f"stays high as the correctness bar tightens", fontsize=11)
        fig.tight_layout()
        out_png = os.path.join(OUT_DIR, "anchor_sweep.png")
        fig.savefig(out_png, dpi=150)
        print(f"Saved plot -> {out_png}")
    except ImportError:
        print("matplotlib not installed; skipped plot (CSV still written).")
    return res


def main():
    df = build_table()
    if DO_SWEEP:
        sweep(df)
    else:
        run_single(df)


if __name__ == "__main__":
    main()































# #!/usr/bin/env python3
# """
# Anchor-based model evaluation with accuracy, ARI, and NMI, plus a threshold sweep.
#
# Idea:
#   - An "anchor" cell is one where the models largely AGREE on the aligned label
#     (the easy cells). Agreement is measured model-to-model on `aligned`.
#   - We evaluate each model against TRUE labels on the NON-anchor (hard) cells.
#     true_label is used ONLY for scoring, never to define the anchor.
#
# Anchor rule:
#   A cell is an anchor if at least a given fraction (percentage) of the consensus
#   models share the same aligned label. The fraction is converted to a required
#   count via ceil, so it behaves consistently for 7 models (common mode) or 6
#   (leave_one_out). Example: 0.80 -> >=6 of 7 (common), >=5 of 6 (loo). 1.0 = unanimous.
#
# Sweep:
#   No single threshold "proves" a model. A strong model STAYS high as the test set
#   is tightened toward the hardest cells. So we sweep the threshold from simple
#   majority up to unanimous and report each model's non-anchor acc/ari/nmi with the
#   non-anchor n at every level. Read the CURVE: the model that stays highest as n
#   shrinks toward the hard cells is the strong one. Ignore tiny gaps at small n.
#
# Metrics (permutation note):
#   ari/nmi are permutation-invariant (same on predicted or aligned); only acc uses
#   your alignment. All three are computed within the subset.
#
# Modes:
#   common       one shared non-anchor set from all models (scores comparable;
#                a model's own votes helped draw the split).
#   leave_one_out anchor from the OTHER models when scoring each model (fair per
#                model; each model's non-anchor set differs, so not strictly
#                comparable across models).
#
# Requires: pandas, numpy, scikit-learn, matplotlib (matplotlib optional; skipped if absent)
# """
#
# import os
# import math
# import pandas as pd
# import numpy as np
# from functools import reduce
# from sklearn.metrics import (accuracy_score,
#                              adjusted_rand_score,
#                              normalized_mutual_info_score)
#
# # ----------------------------- CONFIG ---------------------------------------
# # MODELS      = ["scAce","scAGCL","scCDCG","scCGNet","scDeepCluster","scMAE","scRCL"]          # Model_1 .. Model_7
# MODELS      = ["scAGCL","scCGNet","scDeepCluster","scMAE","scRCL"]          # Model_1 .. Model_7
# FILE_TMPL   = "{model}.csv"          # e.g. "Model_1.csv"; change ext if needed
# READ_KW     = dict()                 # e.g. dict(sep="\t") for TSV
# COL_ID      = "cell_id"
# COL_ALIGNED = "pred_label_aligned"
# COL_TRUE    = "true_label"
# dataset_name = "Lung"
# MODE           = "common"            # "common" or "leave_one_out"
# DO_SWEEP       = True                # sweep threshold from majority -> unanimous
# SWEEP_MIN_FRAC = 0.50                # lowest fraction to start the sweep at
# OUT_DIR        = f"Output/{dataset_name}"                 # where results CSV and plot are written
# os.makedirs(OUT_DIR, exist_ok=True)
# # ----------------------------------------------------------------------------
#
#
# def threshold_count(frac, n):
#     """Smallest integer c such that c/n >= frac ('at least frac%')."""
#     c = int(math.ceil(frac * n - 1e-9))
#     return max(1, min(c, n))
#
#
# def load_model(model):
#     path = f"cluster_data/{dataset_name}/{FILE_TMPL.format(model=model)}"
#     df = pd.read_csv(path, **READ_KW)
#     missing = {COL_ID, COL_ALIGNED, COL_TRUE} - set(df.columns)
#     if missing:
#         raise ValueError(f"{path} is missing columns: {missing}")
#     out = df[[COL_ID, COL_ALIGNED, COL_TRUE]].copy()
#     out = out.rename(columns={COL_ALIGNED: f"aligned_{model}",
#                               COL_TRUE:    f"true_{model}"})
#     return out
#
#
# def build_table():
#     frames = [load_model(m) for m in MODELS]
#     sizes = {m: len(f) for m, f in zip(MODELS, frames)}
#     merged = reduce(lambda a, b: a.merge(b, on=COL_ID, how="inner"), frames)
#     print("Rows per file:", sizes)
#     print(f"Rows after inner join on {COL_ID}: {len(merged)}")
#     dropped = {m: sizes[m] - len(merged) for m in MODELS}
#     if any(v > 0 for v in dropped.values()):
#         print("WARNING: join dropped cells (id mismatch across files):", dropped)
#
#     true_cols = [f"true_{m}" for m in MODELS]
#     consistent = merged[true_cols].nunique(axis=1).eq(1)
#     if not consistent.all():
#         n_bad = int((~consistent).sum())
#         raise ValueError(f"{n_bad} cells have DIFFERENT true_label across files.")
#     merged["true_label"] = merged[true_cols[0]]
#     return merged.drop(columns=true_cols)
#
#
# def agreement_count(df, models):
#     cols = [f"aligned_{m}" for m in models]
#     sub = df[cols]
#     modal = sub.mode(axis=1)[0]
#     count = sub.eq(modal, axis=0).sum(axis=1)
#     return count, modal
#
#
# def _metrics(df, mask, model):
#     sub = df[mask]
#     if len(sub) == 0:
#         return np.nan, np.nan, np.nan
#     yt, yp = sub["true_label"], sub[f"aligned_{model}"]
#     return (accuracy_score(yt, yp),
#             adjusted_rand_score(yt, yp),
#             normalized_mutual_info_score(yt, yp))
#
#
# def metrics_on_subset(df, mask, models, label):
#     rows = []
#     for m in models:
#         acc, ari, nmi = _metrics(df, mask, m)
#         rows.append({"model": m, "subset": label, "n": int(mask.sum()),
#                      "acc": acc, "ari": ari, "nmi": nmi})
#     return pd.DataFrame(rows)
#
#
# # --------------------------- single-threshold run ---------------------------
# def run_single(df):
#     if MODE == "common":
#         n = len(MODELS)
#         thr = threshold_count(SWEEP_MIN_FRAC, n)  # reuse min frac as the single point
#         count, _ = agreement_count(df, MODELS)
#         anchor = count >= thr
#         print(f"\n[common] need >= {thr}/{n} agree  "
#               f"anchor={int(anchor.sum())}  non-anchor={int((~anchor).sum())}")
#         for lab, mask in [("NON-ANCHOR (hard)", ~anchor),
#                           ("ANCHOR (easy)", anchor),
#                           ("OVERALL", pd.Series(True, index=df.index))]:
#             t = metrics_on_subset(df, mask, MODELS, lab)
#             print(f"\n=== {lab} ===")
#             print(t[["model", "n", "acc", "ari", "nmi"]].to_string(index=False))
#     else:
#         rows = []
#         for m in MODELS:
#             others = [x for x in MODELS if x != m]
#             thr = threshold_count(SWEEP_MIN_FRAC, len(others))
#             count, _ = agreement_count(df, others)
#             acc, ari, nmi = _metrics(df, ~(count >= thr), m)
#             rows.append({"model": m, "thr": thr, "n": int((~(count >= thr)).sum()),
#                          "acc": acc, "ari": ari, "nmi": nmi})
#         print(pd.DataFrame(rows).sort_values("acc", ascending=False)
#               .to_string(index=False))
#
#
# # ------------------------------- sweep --------------------------------------
# def sweep(df):
#     records = []
#     if MODE == "common":
#         n = len(MODELS)
#         lo = threshold_count(SWEEP_MIN_FRAC, n)
#         count, _ = agreement_count(df, MODELS)
#         for thr in range(lo, n + 1):
#             non = ~(count >= thr)
#             for m in MODELS:
#                 acc, ari, nmi = _metrics(df, non, m)
#                 records.append({"thr": thr, "pct": thr / n, "model": m,
#                                 "n_non_anchor": int(non.sum()),
#                                 "acc": acc, "ari": ari, "nmi": nmi})
#     else:  # leave_one_out
#         for m in MODELS:
#             others = [x for x in MODELS if x != m]
#             n = len(others)
#             lo = threshold_count(SWEEP_MIN_FRAC, n)
#             count, _ = agreement_count(df, others)
#             for thr in range(lo, n + 1):
#                 non = ~(count >= thr)
#                 acc, ari, nmi = _metrics(df, non, m)
#                 records.append({"thr": thr, "pct": thr / n, "model": m,
#                                 "n_non_anchor": int(non.sum()),
#                                 "acc": acc, "ari": ari, "nmi": nmi})
#     res = pd.DataFrame(records)
#     res["pct_label"] = res["thr"].astype(str) + "/" + \
#         res.apply(lambda r: str(round(r["thr"] / r["pct"])), axis=1) + \
#         " (" + (res["pct"] * 100).round(0).astype(int).astype(str) + "%)"
#
#     # Printed pivots
#     for metric in ["acc", "ari", "nmi"]:
#         piv = res.pivot_table(index="pct_label", columns="model", values=metric)
#         print(f"\n=== NON-ANCHOR {metric.upper()} vs threshold (rows = agreement level) ===")
#         print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
#     npiv = res.pivot_table(index="pct_label", columns="model", values="n_non_anchor")
#     print("\n=== non-anchor n at each threshold ===")
#     print(npiv.astype("Int64").to_string())
#
#     out_csv = os.path.join(OUT_DIR, "anchor_sweep_results.csv")
#     res.to_csv(out_csv, index=False)
#     print(f"\nSaved tidy results -> {out_csv}")
#
#     # Plot (optional)
#     try:
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
#         xcol = "pct"
#         for ax, metric in zip(axes, ["acc", "ari", "nmi"]):
#             for m in MODELS:
#                 d = res[res.model == m].sort_values(xcol)
#                 ax.plot(d[xcol] * 100, d[metric], marker="o", label=m)
#             ax.set_title(f"non-anchor {metric.upper()}")
#             ax.set_xlabel("agreement threshold (%)")
#             ax.grid(alpha=0.3)
#         axes[0].set_ylabel("score")
#         # annotate non-anchor n (same across models in common mode) on acc panel
#         if MODE == "common":
#             d0 = res[res.model == MODELS[0]].sort_values(xcol)
#             for _, r in d0.iterrows():
#                 axes[0].annotate(f"n={int(r['n_non_anchor'])}",
#                                  (r[xcol] * 100, r["acc"]),
#                                  textcoords="offset points", xytext=(0, 6),
#                                  fontsize=7, ha="center")
#         axes[-1].legend(fontsize=7, loc="best")
#         fig.suptitle(f"Non-anchor performance sweep ({MODE}) — a strong model "
#                      f"stays high as threshold tightens", fontsize=11)
#         fig.tight_layout()
#         out_png = os.path.join(OUT_DIR, "anchor_sweep.png")
#         fig.savefig(out_png, dpi=150)
#         print(f"Saved plot -> {out_png}")
#     except ImportError:
#         print("matplotlib not installed; skipped plot (CSV still written).")
#     return res
#
#
# def main():
#     df = build_table()
#     if DO_SWEEP:
#         sweep(df)
#     else:
#         run_single(df)
#
#
# if __name__ == "__main__":
#     main()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
