"""bench_utils.py - shared helpers for the urologic spatial-domain benchmark."""
import os, glob
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture

BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
PROC = os.path.join(BASE, "data/processed")
RUNS = os.path.join(BASE, "results/runs")

# labels that are never scored (pathologist asked to exclude them / missing)
EXCLUDE_LABELS = {"Exclude", "nan", "NaN", "None", ""}
MIN_CLASS_SPOTS = 20      # a class must have at least this many spots to count towards K


def list_slides(pattern=""):
    return sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(PROC, "*.h5ad")) if pattern in os.path.basename(f))


def load_slide(sid):
    a = ad.read_h5ad(os.path.join(PROC, sid + ".h5ad"))
    a.var_names_make_unique()
    tot = np.asarray(a.X.sum(1)).ravel()
    a = a[tot > 0].copy()                      # spots with zero counts cannot be normalised
    return a


def scored_labels(a):
    """Series of ground-truth labels with unscored entries set to NaN (index = spots)."""
    if "label" not in a.obs:
        return None
    lab = a.obs["label"].astype(object)
    lab = lab.where(~lab.astype(str).isin(EXCLUDE_LABELS))
    lab = lab.where(lab.notna())
    return lab


def choose_k(a):
    """K = number of annotated classes with >= MIN_CLASS_SPOTS scored spots (None when no usable labels)."""
    lab = scored_labels(a)
    if lab is None:
        return None
    vc = lab.dropna().value_counts()
    k = int((vc >= MIN_CLASS_SPOTS).sum())
    return k if k >= 2 else None


def nn_dist(xy):
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def lattice_neighbors(xy, factor=1.1):
    """List of neighbour index arrays: spots within factor * nearest-neighbour distance (6 on a full hex grid)."""
    d = nn_dist(xy)
    tree = cKDTree(xy)
    nb = tree.query_ball_point(xy, r=factor * d)
    return [np.array([j for j in n if j != i], dtype=int) for i, n in enumerate(nb)], d


def gmm_tied(Z, k, seed=0):
    """Gaussian mixture with one shared full covariance (the sklearn analogue of mclust 'EEE')."""
    Z = np.asarray(Z, dtype=np.float64)
    gm = GaussianMixture(n_components=k, covariance_type="tied", n_init=5, reg_covar=1e-4, random_state=seed, max_iter=300)
    return gm.fit_predict(Z)


def leiden_k(a, k, use_rep="X_pca", seed=0, n_neighbors=15):
    """Leiden clustering with resolution searched (bisection) to give k clusters; returns closest if not exact."""
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep=use_rep, random_state=seed)
    lo, hi, best = 0.02, 3.0, None
    for _ in range(25):
        mid = (lo + hi) / 2
        sc.tl.leiden(a, resolution=mid, random_state=seed, key_added="_ld", flavor="igraph", n_iterations=2, directed=False)
        n = a.obs["_ld"].nunique()
        if best is None or abs(n - k) < abs(best[0] - k):
            best = (n, a.obs["_ld"].astype(int).values.copy(), mid)
        if n == k:
            break
        if n < k:
            lo = mid
        else:
            hi = mid
    return best[1], best[2]


def standard_pca(a, n_hvg=3000, n_pcs=30, seed=0):
    """Common preprocessing for the non-spatial baselines: HVG (seurat_v3 on counts) -> CP10K -> log1p -> scale -> PCA."""
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=min(n_hvg, b.n_vars))
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    b = b[:, b.var["highly_variable"]].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=n_pcs, random_state=seed)
    return b
