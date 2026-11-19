"""63_dose_generic.py - PCA-smooth dose-response on any set of sections, with label-free marker-programme scores.

Same smoothing as 33_dose_smooth.py: Z_s = (1 - alpha) Z + alpha * mean of Z over the lattice ring(s), then the
tied-covariance GMM with K fixed. K = annotated classes (>= 20 spots) when labels exist, otherwise K = 6
(the K used for the unlabelled sections in the benchmark); ccRCC uses K = 6 as in the TLS analysis.
Two feature sets: 'all' (3,000 HVGs) and 'heldout' (the same HVGs minus every marker gene of the tissue), so that
the clustering cannot see the genes that define the programmes.
Per fit: ARI / NMI (if labels), PAS, and eta^2 of every marker programme.
Per section: programme grain (boundary share of top-decile spots, Moran's I) -> grain_<slide>.csv
Usage: python 63_dose_generic.py <pattern[,pattern]> <out_name> <n_procs> [rings, e.g. 1 or 1,2,3] [seeds, e.g. 0,1,2]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu
import marker_programs as mp

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
RING_F = {1: 1.1, 2: 2.1, 3: 3.1}
OUT, RINGS, SEEDS = None, [1], [0, 1, 2]


def pas(xy, lab, k=10):
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    return float(((lab[idx[:, 1:]] != lab[:, None]).sum(1) > 6).mean())


def pcs(a, exclude=(), n_hvg=3000, n_pcs=30, seed=0):
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=min(n_hvg, b.n_vars))
    if exclude:
        b.var.loc[b.var_names.isin(list(exclude)), "highly_variable"] = False
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    b = b[:, b.var["highly_variable"]].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=n_pcs, random_state=seed)
    return np.asarray(b.obsm["X_pca"], dtype=np.float64)


def one_slide(sid):
    od = os.path.join(bu.BASE, "results/dose", OUT)
    out_csv = os.path.join(od, sid + ".csv")
    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    a = bu.load_slide(sid)
    lab = bu.scored_labels(a)
    tissue, progs = mp.programs_for(sid)
    k = 6 if ("GSE175540" in sid or lab is None) else bu.choose_k(a)
    if not k:
        k = 6
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    scores = mp.program_scores(a, progs) if progs else {}
    nb1, _ = bu.lattice_neighbors(xy, 1.1)
    grain = []
    for p, (s, g) in scores.items():
        tb, mo = mp.grain(s, nb1)
        grain.append(dict(slide=sid, tissue=tissue, program=p, n_genes=len(g), genes=" ".join(g), top_boundary=tb, moran=mo))
    os.makedirs(od, exist_ok=True)
    if grain:
        pd.DataFrame(grain).to_csv(os.path.join(od, f"grain_{sid}.csv"), index=False)
    ok = lab.notna().values if lab is not None else None
    yt = lab[ok].astype(str).values if lab is not None else None
    variants = {"all": pcs(a)}
    if progs:
        allg = sorted({g for v in progs.values() for g in v})
        variants["heldout"] = pcs(a, exclude=allg)
    rows = []
    for var, Z in variants.items():
        for ring in RINGS:
            nb, _ = bu.lattice_neighbors(xy, RING_F[ring])
            n = len(nb)
            r = np.concatenate([np.full(len(v), i, dtype=int) for i, v in enumerate(nb)])
            c = np.concatenate(nb).astype(int)
            W = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
            deg = np.asarray(W.sum(1)).ravel()
            WZ = np.asarray(sp.diags(1.0 / np.maximum(deg, 1)) @ W @ Z)
            WZ[deg == 0] = Z[deg == 0]
            for alpha in ALPHAS:
                if alpha == 0 and ring != RINGS[0]:
                    continue
                Zs = (1 - alpha) * Z + alpha * WZ
                for seed in SEEDS:
                    pred = bu.gmm_tied(Zs, k, seed)
                    row = dict(slide=sid, tissue=tissue, variant=var, k=k, ring=ring if alpha > 0 else 0, alpha=alpha,
                               seed=seed, n_found=int(len(np.unique(pred))), pas=pas(xy, pred))
                    if lab is not None:
                        row.update(ari=adjusted_rand_score(yt, pred[ok]), nmi=normalized_mutual_info_score(yt, pred[ok]))
                    for p, (s, _) in scores.items():
                        row["eta2|" + p] = mp.eta2(s, pred)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"{sid[-34:]:>34} {str(tissue):>8} k={k}: {len(df)} fits, {len(scores)} programmes", flush=True)
    return df


def init(out, rings, seeds):
    global OUT, RINGS, SEEDS
    OUT, RINGS, SEEDS = out, rings, seeds


if __name__ == "__main__":
    pats = sys.argv[1].split(","); out = sys.argv[2]; nproc = int(sys.argv[3])
    rings = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1]
    seeds = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1, 2]
    init(out, rings, seeds)
    slides = [s for s in bu.list_slides() if any(p in s for p in pats)]
    print(f"sections: {len(slides)} | rings {rings} | seeds {seeds} | procs {nproc}", flush=True)
    with Pool(nproc, initializer=init, initargs=(out, rings, seeds)) as p:
        res = list(p.imap_unordered(one_slide, slides))
    D = pd.concat([r for r in res if r is not None], ignore_index=True)
    D.to_csv(os.path.join(bu.BASE, "results/dose", out + "_dose.csv"), index=False)
    G = [pd.read_csv(f) for f in [os.path.join(bu.BASE, "results/dose", out, f"grain_{s}.csv") for s in slides] if os.path.exists(f)]
    if G:
        pd.concat(G, ignore_index=True).to_csv(os.path.join(bu.BASE, "results/dose", out + "_grain.csv"), index=False)
    print("DOSE_GENERIC_DONE", flush=True)
