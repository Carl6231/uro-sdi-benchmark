"""33_dose_smooth.py
Dose-response of spatial smoothing with everything else fixed (same 30 PCs, same tied-covariance GMM, same K):
    Z_s(i) = (1 - alpha) * Z(i) + alpha * mean of Z over the neighbourhood N_r(i)
alpha = 0 is the non-spatial GMM baseline; alpha = 0.5 with r = 1 ring is the benchmark's PCA-smooth.
Neighbourhoods are rings of the Visium hexagonal lattice: r = 1 (6 spots), 2 (18 spots), 3 (36 spots).
Metrics per fit: ARI / NMI against the reference labels, PAS of the prediction, prostate coarse ARI and tumour F1,
ccRCC TLS best-cluster F1 (K = 6, sections with >= 20 TLS spots).
Output: results/dose/smooth_dose.csv (+ one CSV per section in results/dose/smooth/)
Usage: python 33_dose_smooth.py <n_procs>
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RINGS = {1: 1.1, 2: 2.1, 3: 3.1}          # neighbourhood radius = factor x spot spacing
SEEDS = [0, 1, 2, 3, 4]
COARSE = {"GG1": "Tumor", "GG2": "Tumor", "GG4": "Tumor", "GG4 Cribriform": "Tumor",
          "Benign": "Benign", "Benign*": "Benign", "Stroma": "Stroma"}
OUTD = os.path.join(bu.BASE, "results/dose")


def dataset_of(sid):
    if "Erickson" in sid: return "prostate"
    if "Maynard" in sid: return "DLPFC"
    if "GSE175540" in sid: return "ccRCC"
    return "other"


def pas(xy, lab, k=10):
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    return float(((lab[idx[:, 1:]] != lab[:, None]).sum(1) > 6).mean())


def best_f1(pred, pos, neg):
    f = 0.0
    for cl in np.unique(pred):
        m = pred == cl
        tp = int((m & pos).sum())
        if tp:
            f = max(f, 2 * tp / (2 * tp + int((m & neg).sum()) + int((~m & pos).sum())))
    return f


def one_slide(sid):
    out_csv = os.path.join(OUTD, "smooth", sid + ".csv")
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    ds = dataset_of(sid)
    a = bu.load_slide(sid)
    lab = bu.scored_labels(a)
    if lab is None:
        return None
    k = 6 if ds == "ccRCC" else bu.choose_k(a)
    ok = lab.notna().values
    pos_tls, neg_tls = (lab == "TLS").values, (lab == "NO_TLS").values
    if not k or (ds == "ccRCC" and (pos_tls.sum() < 20 or neg_tls.sum() == 0)):
        return None
    yt = lab[ok].astype(str).values
    co = lab.map(COARSE); okc = co.notna().values
    tum = (co == "Tumor").values
    Z = np.asarray(bu.standard_pca(a, seed=0).obsm["X_pca"], dtype=np.float64)
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    rows = []
    for ring, fac in RINGS.items():
        nb, _ = bu.lattice_neighbors(xy, fac)
        n = len(nb)
        r = np.concatenate([np.full(len(v), i, dtype=int) for i, v in enumerate(nb)])
        c = np.concatenate(nb).astype(int)
        W = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        deg = np.asarray(W.sum(1)).ravel()
        WZ = np.asarray(sp.diags(1.0 / np.maximum(deg, 1)) @ W @ Z)
        WZ[deg == 0] = Z[deg == 0]                 # isolated spots keep their own profile
        for alpha in ALPHAS:
            if alpha == 0 and ring > 1:
                continue                           # alpha = 0 does not depend on the neighbourhood
            Zs = (1 - alpha) * Z + alpha * WZ
            for seed in SEEDS:
                pred = bu.gmm_tied(Zs, k, seed)
                row = dict(slide=sid, dataset=ds, k=k, ring=ring if alpha > 0 else 0,
                           mean_neighbors=float(deg.mean()) if alpha > 0 else 0.0, alpha=alpha, seed=seed,
                           n_found=int(len(np.unique(pred))), ari=adjusted_rand_score(yt, pred[ok]),
                           nmi=normalized_mutual_info_score(yt, pred[ok]), pas=pas(xy, pred))
                if okc.sum() and co[okc].nunique() >= 2:
                    row["ari_coarse"] = adjusted_rand_score(co[okc].values, pred[okc])
                    if tum.sum() >= 20:
                        row["tumor_f1"] = best_f1(pred, tum, ok & ~tum)
                if ds == "ccRCC":
                    row["tls_f1"] = best_f1(pred, pos_tls, neg_tls)
                rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"{sid[-30:]:>30} {ds:>8} k={k}: {len(df)} fits", flush=True)
    return df


if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    order = {"prostate": 0, "DLPFC": 1, "ccRCC": 2}
    slides = sorted([s for s in bu.list_slides() if dataset_of(s) in order], key=lambda s: (order[dataset_of(s)], s))
    print(f"sections: {len(slides)} | alpha {ALPHAS} | rings {list(RINGS)} | seeds {SEEDS} | processes {nproc}", flush=True)
    with Pool(nproc) as p:
        res = list(p.imap_unordered(one_slide, slides))
    D = pd.concat([r for r in res if r is not None], ignore_index=True)
    D.to_csv(os.path.join(OUTD, "smooth_dose.csv"), index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.3f}".format)
    for ds, met in [("DLPFC", "ari"), ("prostate", "ari"), ("prostate", "tumor_f1"), ("ccRCC", "tls_f1")]:
        if met not in D:
            continue
        d = D[(D["dataset"] == ds) & D[met].notna()]
        if d.empty:
            continue
        t = d.groupby(["slide", "ring", "alpha"])[met].mean().groupby(["ring", "alpha"]).mean().unstack("alpha")
        print(f"\n== {ds} {met}: mean over seeds, then sections (n = {d['slide'].nunique()}); ring 0 = no smoothing ==\n{t.to_string()}")
    print("DOSE_SMOOTH_DONE", flush=True)
