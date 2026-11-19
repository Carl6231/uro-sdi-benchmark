"""74_marker_grain.py - spatial grain of every marker programme in every real section (label-free):
boundary share and spot-weighted patch size of the top-decile spots, Moran's I (first ring), share of spots with
programme score > 1 SD. Output: results/tables/marker_grain_all.csv
Usage: python 74_marker_grain.py [n_procs]
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu
import marker_programs as mp

PATS = ["Erickson", "Maynard", "GSE175540", "GSE183456", "GSE206306", "GSE226533", "Human_Prostate_10X", "Human_Kidney_10X",
        "GSE213688", "Human_Breast_10X_06092021_Visium", "Human_Breast_10X_06232020_Visium_Block_A_Section_1", "WU2021_"]


def one(sid):
    tissue, progs = mp.programs_for(sid)
    a = bu.load_slide(sid)
    sc = mp.program_scores(a, progs)
    nb, _ = bu.lattice_neighbors(np.asarray(a.obsm["spatial"], float), 1.1)
    rows = []
    for p, (s, g) in sc.items():
        tb, mo = mp.grain(s, nb)
        z = (s - s.mean()) / s.std()
        rows.append(dict(slide=sid, tissue=tissue, program=p, n_genes=len(g), genes=" ".join(g), top_boundary=tb, moran=mo,
                         top_patch=mp.top_patch(s, nb), frac_high=float((z > 1).mean()), n_spots=a.n_obs))
    print(sid[-34:], tissue, len(rows), flush=True)
    return rows


if __name__ == "__main__":
    slides = [s for s in bu.list_slides() if any(p in s for p in PATS)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        res = pool.map(one, slides)
    pd.DataFrame([r for rr in res for r in rr]).to_csv(os.path.join(bu.BASE, "results/tables/marker_grain_all.csv"), index=False)
    print("GRAIN_DONE")
