"""External renal TLS pilot on the three Zurich Visium samples (Zenodo 14620362).

Set up inputs/zurich/KC{1,2,3}.h5ad from the public h5ad_preprocessed folder.
The shared benchmark helpers are in the repository's scripts directory.
Use --plot-only to redraw saved calls, or --shuffle-control for the secondary
degree-preserving adjacency control at alpha=0.7.
"""
import argparse
import importlib
import os
from pathlib import Path
import sys
import warnings

# The shared dose module uses one Numba thread. Set this before Scanpy initializes Numba.
os.environ["NUMBA_NUM_THREADS"] = "1"

import anndata as ad
from anndata.io import read_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "scripts"))
import bench_utils as bench
import marker_programs as markers

dose = importlib.import_module("63_dose_generic")
SAMPLES = ("KC1", "KC2", "KC3")
ALPHAS = (0.0, 0.3, 0.7)
SEEDS = (0, 1, 2)
PROGRAM = "TLS (B cells, FDC)"
MARKERS = markers.CCRCC[PROGRAM]
NEGATIVE = ("INFL", "NOR", "TUM")
RUNS = BASE / "zurich_tls_runs.csv"
CALLS = BASE / "zurich_tls_spots.csv.gz"


def source(sample):
    return BASE / "inputs" / "zurich" / (sample + ".h5ad")


def load_sample(sample):
    # The public dense X contains counts. Read blocks and skip the high-resolution
    # image until plotting so the three specimens never share a full dense matrix.
    with h5py.File(source(sample)) as h:
        obs = read_elem(h["obs"])
        var = read_elem(h["var"])
        xy = h["obsm/spatial"][:]
        blocks = []
        for start in range(0, len(obs), 128):
            block = h["X"][start:start + 128]
            if not (np.isfinite(block).all() and np.min(block) >= 0 and np.array_equal(block, np.rint(block))):
                raise ValueError(f"{sample} X contains non-count values")
            blocks.append(sparse.csr_matrix(block.astype(np.float32)))
    a = ad.AnnData(X=sparse.vstack(blocks, format="csr"), obs=obs, var=var, obsm={"spatial": xy})
    a.var_names_make_unique()
    keep = np.asarray(a.X.sum(1)).ravel() > 0
    assert a.obs_names.is_unique and a.obs["in_tissue"].eq(1).all()
    return a[keep].copy()


def first_ring(xy):
    neighbours, _ = bench.lattice_neighbors(xy, factor=1.1)
    degree = np.fromiter(map(len, neighbours), dtype=int)
    row = np.repeat(np.arange(len(xy)), degree)
    col = np.concatenate(neighbours)
    graph = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(xy), len(xy)))
    assert degree.max() <= 6 and (graph != graph.T).nnz == 0
    return graph, degree


def component_ids(graph, mask):
    ids = np.full(mask.size, -1, dtype=int)
    if np.any(mask):
        _, ids[mask] = connected_components(graph[mask][:, mask], directed=False)
    return ids


def patch_matches(graph, reference, candidate):
    # Masks have already excluded unassigned tissue. One-to-one matching first
    # maximizes the number reaching IoU >= 0.25, then breaks ties using IoU.
    ref = component_ids(graph, reference)
    pred = component_ids(graph, candidate)
    nr, npred = int(ref.max()) + 1, int(pred.max()) + 1
    rsize = np.bincount(ref[ref >= 0], minlength=nr)
    psize = np.bincount(pred[pred >= 0], minlength=npred)
    hits = np.zeros(nr, dtype=bool)
    if nr and npred:
        both = (ref >= 0) & (pred >= 0)
        inter = np.zeros((nr, npred), dtype=int)
        np.add.at(inter, (ref[both], pred[both]), 1)
        iou = inter / (rsize[:, None] + psize[None, :] - inter)
        qualifies = iou >= 0.25
        weights = qualifies * (1 + iou / (1 + min(nr, npred)))
        ii, jj = linear_sum_assignment(weights, maximize=True)
        hits[ii] = qualifies[ii, jj]
    return {"ref_patches": nr, "pred_patches_scored_area": npred, "matched_patches": int(hits.sum()),
            "ref_patches_lt20": int((rsize < 20).sum()), "matched_patches_lt20": int(hits[rsize < 20].sum())}


def representation(a):
    score, present = markers.program_scores(a, {PROGRAM: MARKERS})[PROGRAM]
    # Exclude the anchor genes from PCA, using the existing benchmark's held-out
    # feature implementation. No histology label or pathological K enters the fit.
    pcs = dose.pcs(a, exclude=MARKERS, seed=0)
    graph, degree = first_ring(a.obsm["spatial"])
    return pcs, score, present, graph, degree


def tls_candidate(clusters, score):
    values = np.unique(clusters)
    tls = values[np.argmax([score[clusters == k].mean() for k in values])]
    return clusters == tls


def analyse(sample):
    a = load_sample(sample)
    pcs, score, present, graph, degree = representation(a)
    xy = a.obsm["spatial"]
    local = np.asarray(graph @ pcs) / np.maximum(degree[:, None], 1)
    local[degree == 0] = pcs[degree == 0]

    labels = a.obs["manual_anno"].astype(str).to_numpy()
    subtypes = a.obs["manual_anno_tls"].astype(str).to_numpy()
    assert set(labels) <= {"TLS", "UNASSIGNED", *NEGATIVE}, (sample, np.unique(labels))
    positive = labels == "TLS"
    negative = np.isin(labels, NEGATIVE)
    scored = positive | negative
    rows = []
    calls = pd.DataFrame({"sample": sample, "spot": a.obs_names, "x": xy[:, 0], "y": xy[:, 1],
                          "manual_anno": labels, "manual_anno_tls": subtypes})
    for alpha in ALPHAS:
        z = (1 - alpha) * pcs + alpha * local
        for seed in SEEDS:
            with warnings.catch_warnings(record=True) as logged:
                warnings.simplefilter("always", ConvergenceWarning)
                clusters = bench.gmm_tied(z, 6, seed)
            pred = tls_candidate(clusters, score)
            calls[f"call_a{alpha:.1f}_s{seed}"] = pred.astype(int)
            tp = int((pred & positive).sum())
            fp = int((pred & negative).sum())
            fn = int((~pred & positive).sum())
            row = {"sample": sample, "alpha": alpha, "seed": seed, "spots": a.n_obs, "scored_spots": int(scored.sum()),
                   "markers_used": ",".join(present), "n_tls": int(positive.sum()), "tp": tp, "fp": fp, "fn": fn,
                   "tls_f1": 2 * tp / (2 * tp + fp + fn), "recall": tp / (tp + fn),
                   "precision": tp / (tp + fp) if tp + fp else np.nan,
                   "label_ari": adjusted_rand_score(labels[scored], clusters[scored]),
                   "gmm_warning": any(issubclass(w.category, ConvergenceWarning) for w in logged)}
            for subtype in ("ETLS", "MTLS", "GC"):
                subset = (subtypes == subtype) & positive
                row["n_" + subtype] = int(subset.sum())
                row["hit_" + subtype] = int((pred & subset).sum())
            row.update(patch_matches(graph, positive, pred & scored))
            rows.append(row)
    return rows, calls


def shuffle_control():
    rows = []
    for sample in SAMPLES:
        a = load_sample(sample)
        pcs, score, _, graph, degree = representation(a)
        labels = a.obs["manual_anno"].astype(str).to_numpy()
        positive, negative = labels == "TLS", np.isin(labels, NEGATIVE)
        for permutation_seed in range(101, 106):
            rng = np.random.default_rng(permutation_seed)
            perm = np.arange(len(degree))
            for d in np.unique(degree):
                idx = np.flatnonzero(degree == d)
                perm[idx] = rng.permutation(idx)
            shuffled = graph[perm][:, perm]
            assert np.array_equal(degree, np.asarray(shuffled.sum(1)).ravel())
            true_edge_fraction = graph.multiply(shuffled).nnz / graph.nnz
            neighbours = np.asarray(shuffled @ pcs) / np.maximum(degree[:, None], 1)
            neighbours[degree == 0] = pcs[degree == 0]
            z = .3 * pcs + .7 * neighbours
            for seed in SEEDS:
                pred = tls_candidate(bench.gmm_tied(z, 6, seed), score)
                tp, fp = int((pred & positive).sum()), int((pred & negative).sum())
                fn = int((~pred & positive).sum())
                row = {"sample": sample, "alpha": .7, "permutation_seed": permutation_seed, "seed": seed,
                       "retained_true_edges": true_edge_fraction, "tp": tp, "fp": fp, "fn": fn,
                       "tls_f1": 2 * tp / (2 * tp + fp + fn)}
                row.update(patch_matches(graph, positive, pred & (positive | negative)))
                rows.append(row)
        print(sample, "adjacency control completed", flush=True)
    pd.DataFrame(rows).to_csv(BASE / "zurich_tls_shuffled_neighbours.csv", index=False)


def write_patch_table(spots):
    # Report exact overlaps for every reference patch, including those with zero
    # detected spots. This endpoint does not depend on the IoU matching threshold.
    rows = []
    for sample, section in spots.groupby("sample", sort=True):
        graph, _ = first_ring(section[["x", "y"]].to_numpy())
        ref = section["manual_anno"].eq("TLS").to_numpy()
        ids = component_ids(graph, ref)
        for patch in np.unique(ids[ref]):
            subset = section.loc[ids == patch]
            row = {"sample": sample, "patch": int(patch) + 1, "size_spots": len(subset),
                   "annotation_subtypes": ",".join(sorted(subset["manual_anno_tls"].unique()))}
            for alpha in ALPHAS:
                for seed in SEEDS:
                    key = f"a{alpha:.1f}_s{seed}"
                    row["hits_" + key] = int(subset["call_" + key].sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(BASE / "zurich_tls_patch_hits.csv", index=False)


def plot_maps():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    spots = pd.read_csv(CALLS)
    runs = pd.read_csv(RUNS)
    fig, axes = plt.subplots(3, 5, figsize=(10.2, 6.9))
    fig.subplots_adjust(left=.055, right=.99, top=.876, bottom=.10, wspace=.045, hspace=.13)
    palette = {"unscored": "#e5e8eb", "other": "#adb9c2", "hit": "#167468", "miss": "#346eaf", "false": "#cc661c"}
    for c, name in enumerate(["H&E", "Reference TLS", "No smoothing", "Weak (α = 0.3)", "Strong (α = 0.7)"]):
        axes[0, c].set_title(name, fontsize=10, pad=7)
    for i, sid in enumerate(SAMPLES):
        d = spots[spots["sample"] == sid]
        x, y = d.x.to_numpy(), d.y.to_numpy()
        ref = d["manual_anno"].eq("TLS").to_numpy()
        known = d["manual_anno"].isin(["TLS", *NEGATIVE]).to_numpy()
        with h5py.File(source(sid)) as h:
            img = h["uns/spatial/library_id/images/hires"][:]
        img_ax = axes[i, 0]
        img_ax.imshow(img, interpolation="bilinear")
        img_ax.text(-.07, .5, sid, transform=img_ax.transAxes, rotation=90,
                    va="center", ha="center", fontsize=11, fontweight="bold")
        for c, ax in enumerate(axes[i]):
            if c:
                ax.scatter(x[~known], y[~known], s=2, c=palette["unscored"], lw=0)
                ax.scatter(x[known], y[known], s=2, c=palette["other"], lw=0)
                if c == 1:
                    ax.scatter(x[ref], y[ref], s=5, c=palette["hit"], lw=0)
                else:
                    alpha = ALPHAS[c - 2]
                    call = d[f"call_a{alpha:.1f}_s0"].to_numpy(dtype=bool)
                    for mask, color, shape in ((known & call & ~ref, "false", "s"), (ref & ~call, "miss", "^"), (ref & call, "hit", "o")):
                        ax.scatter(x[mask], y[mask], s=5, c=palette[color], marker=shape, lw=0)
                    metric = runs[(runs["sample"] == sid) & (runs.alpha == alpha) & (runs.seed == 0)]["tls_f1"].item()
                    ax.text(.02, .02, f"F1 {metric:.2f}", transform=ax.transAxes, fontsize=8,
                            bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=1))
            # obsm coordinates refer to the archived hires image. Use identical
            # limits for histology, reference, and estimates within each sample.
            pad = max(np.ptp(x), np.ptp(y)) * .035
            ax.set(xlim=(x.min()-pad, x.max()+pad), ylim=(y.max()+pad,y.min()-pad), aspect="equal")
            ax.axis("off")
    keys = [("hit","o","Reference / correctly called TLS"), ("miss","^","Missed TLS"),
            ("false","s","False positive"), ("other","o","Scored non-TLS"), ("unscored","o","Unassigned")]
    handles = [Line2D([], [], marker=shape, ls="", color=palette[key], label=label, ms=6) for key,shape,label in keys]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(.5,.025))
    fig.suptitle("TLS calls in three kidney tumour sections", fontsize=16, y=.984)
    fig.text(.5, .933, "K = 6     |     Seed 0 for all maps     |     Scored within manually annotated tissue",
             ha="center", va="center", fontsize=9, color="#435262")
    fig.savefig(BASE / "zurich_tls_maps.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true")
    mode.add_argument("--shuffle-control", action="store_true")
    args = ap.parse_args()
    if args.shuffle_control:
        with threadpool_limits(limits=2):
            shuffle_control()
        sys.exit(0)
    if not args.plot_only:
        all_rows, all_spots = [], []
        with threadpool_limits(limits=2):
            for sample in SAMPLES:
                rows, calls = analyse(sample)
                all_rows.extend(rows)
                all_spots.append(calls)
                print(sample, "completed", flush=True)
        pd.DataFrame(all_rows).to_csv(RUNS, index=False)
        spots = pd.concat(all_spots, ignore_index=True)
        spots.to_csv(CALLS, index=False)
        write_patch_table(spots)
    plot_maps()
