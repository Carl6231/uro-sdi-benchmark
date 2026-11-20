"""Split/merge errors on first-ring spot pairs in labelled Visium sections.

Run from the checkout with URO_SDI_BASE pointing to the original benchmark work area.
The 'archive' mode reads stored predictions. The dose modes reconstruct three
previously scored PCA-smooth weights. 'paired' tracks individual edges at fixed seed.
"""
import argparse
import importlib
import os
from pathlib import Path
import sys

# 63_dose_generic uses a single thread for these libraries.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

import bench_utils as bu
import geom_utils as geometry

BASE = Path(bu.BASE)
TABLES = BASE / "results/tables"
ALPHAS = (0., .3, .7)


def dataset_and_unit(slide):
    if "Maynard" in slide:
        number = int(slide.rsplit("_", 1)[1])
        for first, last in ((151507, 151510), (151669, 151672), (151673, 151676)):
            if first <= number <= last:
                return "DLPFC", f"Maynard_{first}_{last}"
        raise ValueError(f"Unmapped DLPFC section {slide}")
    if "Erickson" in slide:
        if "Patient_1_" not in slide:
            raise ValueError(f"Unmapped prostatectomy {slide}")
        return "prostate", "Erickson_patient_1"
    return "breast", ""  # Do not infer donor IDs from section IDs.


def graph_and_edges(xy):
    _, row, col = geometry.ring_edges(xy, 1.1)
    graph = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(xy), len(xy)))
    if (graph != graph.T).nnz or np.any(row == col):
        raise ValueError("First ring must be symmetric and exclude self")
    unique = row < col
    return graph, row[unique], col[unique]


def edge_counts(ref, pred, left, right, eligible):
    usable = eligible[left] & eligible[right]
    i, j = left[usable], right[usable]
    same = ref[i] == ref[j]
    joined = pred[i] == pred[j]
    within = int(same.sum())
    across = int((~same).sum())
    splits = int((same & ~joined).sum())
    merges = int((~same & joined).sum())
    if not (within and across):
        raise ValueError("Both types of reference edges are required")
    return dict(within_edges=within, boundary_edges=across, false_splits=splits,
                false_merges=merges, split_rate=splits/within, merge_rate=merges/across)


def archive_section(slide, scores, work):
    label, xy, ids = geometry.load_labels(slide)
    ref = label.fillna("").to_numpy()
    valid = label.notna().to_numpy()
    _, left, right = graph_and_edges(xy)
    dataset, unit = dataset_and_unit(slide)
    methods = scores["method"].unique()
    common = valid.copy()
    predictions = []
    for rec in scores.itertuples(index=False):
        f = work / slide / f"{rec.method}_k{int(rec.k)}_s{int(rec.seed)}.csv"
        call = pd.read_csv(f, usecols=["spot", "cluster"]).set_index("spot")["cluster"]
        if not call.index.is_unique or len(call.index.difference(ids)):
            raise ValueError(f"Unexpected barcodes in {f}")
        pred, _ = pd.factorize(call.reindex(ids),sort=False)
        observed = valid & (pred >= 0)
        ari = adjusted_rand_score(ref[observed], pred[observed])
        if not np.isclose(ari, rec.ari, rtol=0, atol=1e-7):
            raise ValueError(f"Saved predictions differ from scored ARI in {f}")
        common &= pred >= 0
        predictions.append((rec, pred, ari))
    if len(methods) != 13 or scores["k"].nunique() != 1:
        raise ValueError(f"Incomplete or mixed-K comparison on {slide}")
    result = []
    for rec, pred, ari in predictions:
        result.append(dict(dataset=dataset, donor=unit, slide=slide, method=rec.method,
                           k=int(rec.k), seed=int(rec.seed), n_reference_spots=int(valid.sum()),
                           n_common_spots=int(common.sum()), ari=ari,
                           **edge_counts(ref, pred, left, right, common)))
    return result


def dose_section(slide, scored, mode):
    dose = importlib.import_module("63_dose_generic")
    a = bu.load_slide(slide)
    labels = bu.scored_labels(a)
    valid = labels.notna().to_numpy()
    ref = labels.fillna("").to_numpy()
    graph, left, right = graph_and_edges(a.obsm["spatial"])
    z = dose.pcs(a)
    deg = np.asarray(graph.sum(1)).ravel()
    neighbour_z = np.asarray(sparse.diags(1. / np.maximum(deg, 1)) @ graph @ z)
    neighbour_z[deg == 0] = z[deg == 0]
    dataset, unit = dataset_and_unit(slide)
    if scored["k"].nunique() != 1 or set(zip(scored.alpha,scored.seed)) != {
            (al, seed) for al in ALPHAS for seed in range(3)} or len(scored) != 9:
        raise ValueError(f"Unexpected dose design for {slide}")
    out = []
    if mode == "breast_pairs":
        border = valid[left] & valid[right] & (ref[left] != ref[right])
        i, j = left[border], right[border]
        classes = np.sort(np.column_stack([ref[i],ref[j]]),axis=1)
        pairs = pd.DataFrame(classes,columns=["class_a","class_b"])
    if mode == "paired":
        usable = valid[left] & valid[right]
        i, j = left[usable], right[usable]
        reference_same = ref[i] == ref[j]
        endpoints = np.sort(np.column_stack((ref[i], ref[j])).astype(str), axis=1)
        class_pairs, group = np.unique(endpoints, axis=0, return_inverse=True)
        group = group.reshape(-1)
        baseline_errors = {}
    for rec in scored.sort_values(["alpha", "seed"]).itertuples(index=False):
        if mode == "breast_pairs" and rec.alpha == .7:
            continue
        pred = bu.gmm_tied((1.-rec.alpha)*z+rec.alpha*neighbour_z,int(rec.k),int(rec.seed))
        ari = adjusted_rand_score(ref[valid],pred[valid])
        if not np.isclose(ari, rec.ari, rtol=0, atol=1e-7):
            raise ValueError(f"Reconstructed ARI differs from archive on {slide}, alpha={rec.alpha}, seed={rec.seed}")
        if mode == "paired":
            wrong = (pred[i] == pred[j]) != reference_same
            if rec.alpha == 0:
                baseline_errors[rec.seed] = wrong
                continue
            # The two bits encode error status before and after smoothing.
            state = 2 * baseline_errors[rec.seed].astype(int) + wrong
            counts = np.bincount(4 * group + state, minlength=4 * len(class_pairs)).reshape(-1, 4)
            for (a_name, b_name), (retained, introduced, corrected, persistent) in zip(class_pairs, counts):
                out.append(dict(dataset=dataset, donor=unit, slide=slide, k=int(rec.k),
                                seed=int(rec.seed), alpha=rec.alpha, class_a=a_name, class_b=b_name,
                                edge_type="within" if a_name == b_name else "boundary",
                                n_edges=retained+introduced+corrected+persistent,
                                retained_correct=retained, introduced=introduced,
                                corrected=corrected, persistent_error=persistent))
            continue
        if mode == "breast_pairs":
            detail = pairs.assign(merged=pred[i]==pred[j]).groupby(["class_a","class_b"]).merged.agg(["size","sum"])
            for (ca,cb),counts in detail.iterrows():
                out.append(dict(slide=slide,class_a=ca,class_b=cb,alpha=rec.alpha,seed=int(rec.seed),
                                n_edges=int(counts["size"]),merged=int(counts["sum"])))
            continue
        out.append(dict(dataset=dataset, donor=unit, slide=slide, method="PCA-smooth",
                        k=int(rec.k), alpha=rec.alpha, seed=int(rec.seed),
                        n_reference_spots=int(valid.sum()), n_common_spots=int(valid.sum()),
                        ari=ari, archived_ari=rec.ari, ari_difference=ari-rec.ari,
                        **edge_counts(ref, pred, left, right, valid)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("archive","dose","paired","breast_pairs"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ref_methods = set(pd.read_csv(TABLES / "rank_transfer.csv")["method"])
    breast = set(pd.read_csv(TABLES / "ext_geometry_slide.csv")["slide"])
    main_scores = pd.read_csv(BASE / "results/scores.csv")
    primary = set(main_scores.loc[main_scores.dataset.isin(["DLPFC","prostate"]), "slide"])
    if args.mode == "archive":
        main_scores = main_scores[main_scores.slide.isin(primary)]
        ext = pd.read_csv(BASE / "results/scores_ext.csv")
        scores = pd.concat([main_scores,ext[ext.slide.isin(breast)]])
        scores = scores[scores.method.isin(ref_methods) & scores.status.eq("ok")]
    else:
        s1 = pd.read_csv(BASE / "results/dose/real_dose.csv")
        s2 = pd.read_csv(BASE / "results/dose/ext_dose.csv")
        scores = pd.concat([s1[s1.slide.isin(primary)],s2[s2.slide.isin(breast)]])
        scores = scores[scores.variant.eq("all") & scores.alpha.isin(ALPHAS)
                        & (((scores.alpha == 0) & (scores.ring == 0))
                           | ((scores.alpha > 0) & (scores.ring == 1)))]
    if set(scores.slide) != primary | breast:
        raise ValueError("Missing eligible sections")
    if args.mode=="breast_pairs":
        w=scores[scores.slide.isin(breast)].groupby(["slide","alpha"]).ari.mean().unstack()
        scores=scores[scores.slide.isin(w.index[w[.3]<w[0.]])]
    rows = []
    for slide, frame in scores.groupby("slide", sort=True):
        path = BASE / ("results/runs_ext" if slide in breast else "results/runs")
        rows.extend(archive_section(slide, frame, path) if args.mode == "archive" else
                    dose_section(slide, frame, args.mode))
        print(args.mode, slide, flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    tmp = args.out.with_suffix(".partial")
    result=pd.DataFrame(rows)
    if args.mode=="dose":
        print("Maximum absolute difference from archived ARI:",result.ari_difference.abs().max())
    if args.mode=="breast_pairs":
        groups=["slide","class_a","class_b","n_edges"]
        mean=result.groupby(groups+["alpha"]).merged.mean().unstack()
        result=mean.rename(columns={0.:"merged_alpha0",.3:"merged_alpha03"}).reset_index()
        result["extra_merged_edges"]=result.merged_alpha03-result.merged_alpha0
        result["delta_merge_rate"]=result.extra_merged_edges/result.n_edges
    result.to_csv(tmp, index=False)
    tmp.replace(args.out)


if __name__ == "__main__":
    main()
