"""16_make_r_jobs.py - write meta/r_jobs.txt: one line per R run = "<slide> <method> <K> <seed>" (same K rule as the Python runner)."""
import os, sys
import numpy as np, pandas as pd, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu


def k_of(sid):
    a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), backed="r")
    obs = a.obs.copy(); a.file.close()
    if "label" not in obs:
        return None
    lab = obs["label"].astype(object)
    lab = lab.where(~lab.astype(str).isin(bu.EXCLUDE_LABELS)).where(lab.notna())
    k = int((lab.dropna().value_counts() >= bu.MIN_CLASS_SPOTS).sum())
    return k if k >= 2 else None


jobs = []
pro = bu.list_slides("Erickson"); dl = bu.list_slides("Maynard"); cc = bu.list_slides("GSE175540")
kp = {s: k_of(s) for s in pro + dl}
for m in ["bayesspace", "banksy"]:
    jobs += [f"{s} {m} {kp[s]} {seed}" for s in pro for seed in range(5)]
    jobs += [f"{s} {m} {kp[s]} 0" for s in dl]
jobs += [f"{s} drsc {kp[s]} {seed}" for s in pro for seed in range(5)]
jobs += [f"{s} drsc {kp[s]} 0" for s in dl]
for m in ["banksy", "bayesspace", "drsc"]:
    jobs += [f"{s} {m} {k} 0" for s in cc for k in (6, 10)]
open(os.path.join(bu.BASE, "meta/r_jobs.txt"), "w").write("\n".join(jobs) + "\n")
print("jobs:", len(jobs)); print("K prostate:", {s[-6:]: k for s, k in kp.items() if s in pro}); print("K DLPFC:", {s[-6:]: k for s, k in kp.items() if s in dl})
