"""77_expression_coherence.py - a label-free, K-free descriptor of how coarse the expression pattern of a section is:
Moran's I (first lattice ring) of each of the top 10 principal components (same 3,000-HVG PCA as the dose analysis),
averaged with weights = variance explained. High values: neighbouring spots look alike (large domains); low values:
neighbours differ (fine-grained tissue or noise). Also the neighbour correlation of the 30-PC profiles.
Output: results/tables/expression_coherence.csv
Usage: python 77_expression_coherence.py [n_procs] [pattern,...]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

PATS = ["Erickson", "Maynard", "GSE175540", "GSE183456", "GSE206306", "GSE226533", "Human_Prostate_10X", "Human_Kidney_10X",
        "GSE213688", "Human_Breast_10X_06092021_Visium", "Human_Breast_10X_06232020_Visium_Block_A_Section_1", "WU2021_", "SIM_"]


def moran(z, W):
    z = z - z.mean()
    return float(len(z) / W.sum() * (z @ (W @ z)) / (z @ z))


def one(sid):
    a = bu.load_slide(sid)
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=min(3000, b.n_vars))
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    b = b[:, b.var["highly_variable"]].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=30, random_state=0)
    Z = np.asarray(b.obsm["X_pca"], float); ev = b.uns["pca"]["variance_ratio"]
    nb, _ = bu.lattice_neighbors(np.asarray(a.obsm["spatial"], float), 1.1)
    r = np.concatenate([np.full(len(v), i) for i, v in enumerate(nb)]); c = np.concatenate(nb).astype(int)
    W = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(a.n_obs, a.n_obs))
    mi = np.array([moran(Z[:, j], W) for j in range(10)])
    deg = np.asarray(W.sum(1)).ravel()
    NZ = np.asarray(sp.diags(1 / np.maximum(deg, 1)) @ W @ Z)
    ok = deg > 0
    zc = Z - Z.mean(0); nc = NZ - NZ.mean(0)
    cor = np.sum(zc[ok] * nc[ok], 1) / (np.linalg.norm(zc[ok], axis=1) * np.linalg.norm(nc[ok], axis=1))
    row = dict(slide=sid, n_spots=a.n_obs, moran_pc_w=float(np.sum(mi * ev[:10]) / ev[:10].sum()), moran_pc1=float(mi[0]),
               moran_pc_mean=float(mi.mean()), nb_corr=float(np.median(cor)), median_umi=float(np.median(np.asarray(a.X.sum(1)).ravel())))
    print(sid[-34:], round(row["moran_pc_w"], 3), round(row["nb_corr"], 3), flush=True)
    return row


if __name__ == "__main__":
    pats = sys.argv[2].split(",") if len(sys.argv) > 2 else PATS
    slides = [s for s in bu.list_slides() if any(p in s for p in pats)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        rows = pool.map(one, slides)
    out = os.path.join(bu.BASE, "results/tables/expression_coherence.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(out):
        old = pd.read_csv(out); new = pd.concat([old[~old["slide"].isin(new["slide"])], new])
    new.to_csv(out, index=False)
    print("COHERENCE_DONE")
