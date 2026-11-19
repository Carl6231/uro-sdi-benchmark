"""45_dose_ksens_stats.py - summary statistics for the dose-response and K-sensitivity experiments.
Inputs : results/dose/smooth_dose.csv (PCA-smooth alpha x ring), results/scores_dose.csv (BANKSY lambda, STAGATE radius),
         results/scores.csv (benchmark runs: STAGATE radius 1.1 and every method at the reference K),
         results/scores_ksens.csv (prostate, reference K -2..+2, seed 0)
Outputs: results/tables/dose_summary.csv, dose_trend_per_section.csv, ksens_summary.csv, ksens_per_slide.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

T = os.path.join(bu.BASE, "results/tables")
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400, "display.float_format", "{:.3f}".format)
SP9 = ["graphst", "stagate", "sedr", "spagcn", "spaceflow", "bayesspace", "banksy", "drsc", "smooth", "bass", "precast"]  # all spatial methods
S = pd.read_csv(os.path.join(bu.BASE, "results/scores.csv")); S = S[S["status"] == "ok"]


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; n = len(x)
    if n < 2:
        return np.nan, np.nan
    h = tdist.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean() - h, x.mean() + h


rows, trend = [], []


def summarise(knob, ring, wide, base_level, met):
    """wide: (dataset, slide) x level table of the metric (mean over seeds)."""
    dl = wide.sub(wide[base_level], axis=0)
    for ds in dl.index.get_level_values(0).unique():
        w, a_ = dl.loc[ds], wide.loc[ds]
        for lev in w.columns:
            lo, hi = ci(w[lev])
            rows.append(dict(knob=knob, ring=ring, dataset=ds, metric=met, level=lev, n=int(w[lev].notna().sum()),
                             mean_abs=float(a_[lev].mean()), mean_delta=float(w[lev].mean()), ci_lo=lo, ci_hi=hi,
                             n_up=int((w[lev] > 1e-9).sum()), n_down=int((w[lev] < -1e-9).sum())))
        for sl, r in a_.iterrows():
            r = r.dropna()
            trend.append(dict(knob=knob, ring=ring, dataset=ds, metric=met, slide=sl, rho_level=spearmanr(r.index.values, r.values)[0],
                              best_level=float(r.idxmax()), gain_at_best=float(r.max() - r[base_level]),
                              delta_at_max_level=float(r.iloc[-1] - r[base_level])))


# ---- 1. PCA-smooth: mixing weight alpha for 1, 2, 3 lattice rings ----
D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
for met in ["ari", "tumor_f1", "tls_f1", "pas"]:
    if met not in D:
        continue
    per = D[D[met].notna()].groupby(["dataset", "slide", "ring", "alpha"])[met].mean().reset_index()
    for ring in [1, 2, 3]:
        sub = pd.concat([per[per["alpha"] == 0].assign(ring=ring), per[per["ring"] == ring]], ignore_index=True)
        summarise("PCA-smooth alpha", ring, sub.pivot_table(index=["dataset", "slide"], columns="alpha", values=met), 0.0, met)

# ---- 2. BANKSY lambda and STAGATE graph radius ----
SD = pd.read_csv(os.path.join(bu.BASE, "results/scores_dose.csv")); SD = SD[SD["status"] == "ok"]
b = SD[SD["method"].str.startswith("banksy-lam")].copy(); b["level"] = b["method"].str.replace("banksy-lam", "").astype(float)
st = SD[SD["method"].str.startswith("stagate-r")].copy(); st["level"] = st["method"].str.replace("stagate-r", "").astype(float)
st0 = S[(S["method"] == "stagate") & S["dataset"].isin(st["dataset"].unique()) & S["seed"].isin(st["seed"].unique())
        & S["slide"].isin(st["slide"].unique())].assign(level=1.1)
st = pd.concat([st0, st], ignore_index=True)
for knob, d, base in [("BANKSY lambda", b, 0.0), ("STAGATE radius", st, 1.1)]:
    for met in ["ari", "tumor_f1", "tls_f1", "pas"]:
        if met not in d or d[met].notna().sum() == 0:
            continue
        wide = d[d[met].notna()].groupby(["dataset", "slide", "level"])[met].mean().unstack("level")
        if base in wide:
            summarise(knob, 0, wide, base, met)

R = pd.DataFrame(rows); TR = pd.DataFrame(trend)
R.to_csv(os.path.join(T, "dose_summary.csv"), index=False); TR.to_csv(os.path.join(T, "dose_trend_per_section.csv"), index=False)
print("===== dose-response: change vs the non-spatial / benchmark setting, mean over sections [95% CI], n up / n down =====")
for (knob, met), g in R.groupby(["knob", "metric"], sort=False):
    if met == "pas":
        continue
    print(f"\n--- {knob} | {met}")
    g = g.assign(txt=g.apply(lambda r: f"{r.mean_delta:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] {r.n_up}/{r.n_down}", axis=1))
    print(g.pivot_table(index=["dataset", "ring"], columns="level", values="txt", aggfunc="first").to_string())
print("\n===== per-section trend: Spearman rho of the metric vs dose, and the best dose =====")
print(TR[TR["metric"] != "pas"].groupby(["knob", "ring", "dataset", "metric"]).agg(
    n=("rho_level", "size"), rho_median=("rho_level", "median"), n_rho_pos=("rho_level", lambda x: int((x > 0).sum())),
    best_level_median=("best_level", "median"), gain_at_best=("gain_at_best", "mean")).to_string())

# ---- 3. K sensitivity on prostate (seed 0) ----
fk = os.path.join(bu.BASE, "results/scores_ksens.csv")
if os.path.exists(fk):
    K = pd.read_csv(fk); K = K[(K["status"] == "ok") & (K["dataset"] == "prostate")]
    ref = S[(S["dataset"] == "prostate") & (S["seed"] == 0)]
    kref = ref.groupby("slide")["k"].first()
    A = pd.concat([ref, K], ignore_index=True)
    A["dk"] = A["k"] - A["slide"].map(kref)
    A = A[A["method"] != "spagcn_refined"]
    out = []
    for met in ["ari", "ari_coarse", "tumor_f1"]:
        per = A[A[met].notna()].groupby(["dk", "slide", "method"])[met].mean().unstack("method")
        for dk, w in per.groupby(level=0):
            w = w.droplevel(0).dropna(axis=1, how="all")
            g = w["gmm"]; sp = [m for m in SP9 if m in w]
            out.append(dict(metric=met, dk=dk, n_slides=len(w), gmm=g.mean(), best_spatial=w[sp].mean().idxmax(),
                            best_spatial_value=w[sp].mean().max(), spatial9_mean=w[sp].mean().mean(),
                            n_methods_above_gmm=int((w[sp].mean() > g.mean()).sum()),
                            gmm_rank=int(w.mean().rank(ascending=False, method="min")["gmm"]), n_methods=w.shape[1]))
        # oracle K: every method gets its best K per section
        best = A[A[met].notna()].groupby(["slide", "method"])[met].max().unstack("method")
        sp = [m for m in SP9 if m in best]
        out.append(dict(metric=met, dk="best K", n_slides=len(best), gmm=best["gmm"].mean(), best_spatial=best[sp].mean().idxmax(),
                        best_spatial_value=best[sp].mean().max(), spatial9_mean=best[sp].mean().mean(),
                        n_methods_above_gmm=int((best[sp].mean() > best["gmm"].mean()).sum()),
                        gmm_rank=int(best.mean().rank(ascending=False, method="min")["gmm"]), n_methods=best.shape[1]))
    KS = pd.DataFrame(out); KS.to_csv(os.path.join(T, "ksens_summary.csv"), index=False)
    A.groupby(["slide", "dk", "method"])[["ari", "ari_coarse", "tumor_f1"]].mean().reset_index().to_csv(
        os.path.join(T, "ksens_per_slide.csv"), index=False)
    print("\n===== prostate: K sensitivity (reference K + dk, seed 0) =====")
    print(KS.to_string(index=False))
    print("\nmean ARI per method and dk:")
    print(A.groupby(["method", "dk"])["ari"].mean().unstack("dk").sort_values(0, ascending=False).to_string())
    print("runs per dk:", A.groupby("dk").size().to_dict())
print("DOSE_KSENS_STATS_DONE")
