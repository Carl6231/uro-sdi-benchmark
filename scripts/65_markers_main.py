"""65_markers_main.py - label-free biological check of the benchmark domains with marker-gene programmes.

For every real section with a marker set (prostate, ccRCC, kidney, DLPFC, breast) and every finished run in RUNS_DIR:
eta^2 of each programme score explained by the predicted domains (between-domain / total variance).
Per section and programme also the grain of the programme (boundary share of top-decile spots, Moran's I).
Outputs: results/tables/<out>_eta.csv and results/tables/<out>_grain.csv
Usage: python 65_markers_main.py [runs_dir=results/runs] [out=marker_main] [n_procs=6]
"""
import os, sys, glob, json
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu
import marker_programs as mp

RUNS_DIR = os.path.join(bu.BASE, sys.argv[1]) if len(sys.argv) > 1 else bu.RUNS
OUT = sys.argv[2] if len(sys.argv) > 2 else "marker_main"
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 6


def one(sdir):
    sid = os.path.basename(sdir)
    tissue, progs = mp.programs_for(sid)
    if not progs or not os.path.exists(os.path.join(bu.PROC, sid + ".h5ad")):
        return None, None
    a = bu.load_slide(sid)
    scores = mp.program_scores(a, progs)
    if not scores:
        return None, None
    nb, _ = bu.lattice_neighbors(np.asarray(a.obsm["spatial"], float), 1.1)
    grain = []
    for p, (s, g) in scores.items():
        tb, mo = mp.grain(s, nb)
        grain.append(dict(slide=sid, tissue=tissue, program=p, n_genes=len(g), genes=" ".join(g), top_boundary=tb, moran=mo))
    pos = pd.Series(np.arange(a.n_obs), index=a.obs_names)
    rows = []
    for jf in sorted(glob.glob(os.path.join(sdir, "*.json"))):
        r = json.load(open(jf))
        if r.get("status") != "ok" or not os.path.exists(jf[:-5] + ".csv"):
            continue
        df = pd.read_csv(jf[:-5] + ".csv")
        idx = pos.reindex(df["spot"].astype(str)).values
        okm = ~pd.isna(idx)
        idx = idx[okm].astype(int)
        for col, mname in [("cluster", r["method"]), ("cluster_refined", r["method"] + "_refined")]:
            if col not in df:
                continue
            lab = df[col].values[okm]
            for p, (s, _) in scores.items():
                rows.append(dict(slide=sid, tissue=tissue, method=mname, k=r["k"], seed=r["seed"], program=p,
                                 eta2=mp.eta2(s[idx], lab)))
    print(f"{sid[-34:]:>34} {tissue:>8}: {len(scores)} programmes, {len(rows)} rows", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(grain)


if __name__ == "__main__":
    dirs = sorted(d for d in glob.glob(os.path.join(RUNS_DIR, "*")) if os.path.isdir(d))
    with Pool(NP) as pool:
        res = list(pool.imap_unordered(one, dirs))
    E = pd.concat([e for e, g in res if e is not None], ignore_index=True)
    G = pd.concat([g for e, g in res if g is not None], ignore_index=True)
    td = os.path.join(bu.BASE, "results/tables")
    E.to_csv(os.path.join(td, OUT + "_eta.csv"), index=False)
    G.to_csv(os.path.join(td, OUT + "_grain.csv"), index=False)
    print(E.groupby(["tissue", "program"]).size().to_string())
    print("MARKERS_DONE", flush=True)
