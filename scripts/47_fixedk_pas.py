"""47_fixedk_pas.py - is the label-free warning sign (patchiness of the non-spatial GMM domains) just a K effect?
Recomputes GMM PAS at the same fixed K (4, 6, 8) on every DLPFC and prostate section and correlates it with the
gain of the 9 spatial methods over GMM (benchmark runs, reference K). Output: results/tables/fixedk_pas.csv
Usage: python 47_fixedk_pas.py <n_procs>
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

KS, SEEDS = [4, 6, 8], [0, 1, 2]
SP9 = ["graphst", "stagate", "sedr", "spagcn", "spaceflow", "bayesspace", "banksy", "drsc", "smooth", "bass", "precast"]   # all spatial methods


def pas(xy, lab, k=10):
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    return float(((lab[idx[:, 1:]] != lab[:, None]).sum(1) > 6).mean())


def one(sid):
    a = bu.load_slide(sid)
    Z = bu.standard_pca(a, seed=0).obsm["X_pca"]; xy = np.asarray(a.obsm["spatial"], float)
    return [dict(slide=sid, k=k, seed=s, pas=pas(xy, bu.gmm_tied(Z, k, s))) for k in KS for s in SEEDS]


if __name__ == "__main__":
    slides = [s for s in bu.list_slides() if "Maynard" in s or "Erickson" in s]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as p:
        R = pd.DataFrame([r for rows in p.map(one, slides) for r in rows])
    S = pd.read_csv(os.path.join(bu.BASE, "results/scores.csv")); S = S[(S["status"] == "ok") & S["slide"].isin(slides)]
    ari = S.groupby(["slide", "method"])["ari"].mean().unstack()
    gain = (ari[[m for m in SP9 if m in ari]].mean(1) - ari["gmm"]).rename("gain9")
    P = R.groupby(["slide", "k"])["pas"].mean().unstack("k").add_prefix("pas_k").join(gain)
    P["dataset"] = ["prostate" if "Erickson" in s else "DLPFC" for s in P.index]
    P.to_csv(os.path.join(bu.BASE, "results/tables/fixedk_pas.csv"))
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(P.groupby("dataset").mean(numeric_only=True).to_string())
    for k in KS:
        r = spearmanr(P[f"pas_k{k}"], P["gain9"]); rd = spearmanr(P.loc[P.dataset == "DLPFC", f"pas_k{k}"], P.loc[P.dataset == "DLPFC", "gain9"])
        print(f"K = {k}: GMM PAS vs spatial gain, all 19: rho {r[0]:.2f} P {r[1]:.3g} | DLPFC only: rho {rd[0]:.2f} P {rd[1]:.3g}")
    print("FIXEDK_DONE")
