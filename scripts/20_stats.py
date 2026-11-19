"""20_stats.py
Statistics on results/scores.csv:
 1) per-dataset method summary (slide-level means over seeds)
 2) each method vs the non-spatial GMM baseline: paired Wilcoxon signed-rank across slides, BH-adjusted
 3) does the DLPFC ranking transfer to prostate?  Spearman correlation of method means
 4) spatial coherence of the pathologist labels themselves (PAS / CHAOS) per dataset
 5) relation between smoothness (PAS) and accuracy (ARI) across methods, per dataset
 6) run-to-run variability over seeds (prostate)
Tables are written to results/tables/.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad
from scipy.stats import wilcoxon, spearmanr
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

TAB = os.path.join(bu.BASE, "results/tables"); os.makedirs(TAB, exist_ok=True)
S = pd.read_csv(os.path.join(bu.BASE, "results/scores.csv"))
ok = S[S["status"] == "ok"].copy()
pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:.3f}".format)
SPATIAL_FREE = {"gmm", "leiden"}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k); q[i] = prev
    return q


def pas(xy, lab, k=10):
    _, idx = cKDTree(xy).query(xy, k=min(k + 1, len(xy)))
    return float(((lab[idx[:, 1:]] != lab[:, None]).sum(1) > 6).mean())


means = {}
for ds in ["DLPFC", "prostate"]:
    d = ok[ok["dataset"] == ds]
    per = d.groupby(["method", "slide"])["ari"].mean().unstack()
    summ = pd.DataFrame({"n_slides": per.notna().sum(1), "mean": per.mean(1), "sd": per.std(1), "median": per.median(1),
                         "min": per.min(1), "max": per.max(1)}).sort_values("mean", ascending=False)
    summ["rank"] = np.arange(1, len(summ) + 1)
    summ.to_csv(os.path.join(TAB, f"summary_ari_{ds}.csv")); per.to_csv(os.path.join(TAB, f"ari_per_slide_{ds}.csv"))
    means[ds] = summ["mean"]
    rows = []
    for m in per.index:
        if m == "gmm":
            continue
        both = per.loc[[m, "gmm"]].dropna(axis=1)
        diff = (both.loc[m] - both.loc["gmm"]).values
        try:
            p = wilcoxon(diff, alternative="two-sided").pvalue
        except Exception:
            p = np.nan
        rows.append(dict(method=m, n=len(diff), mean_diff=diff.mean(), median_diff=np.median(diff),
                         n_better=int((diff > 0).sum()), n_worse=int((diff < 0).sum()), p=p))
    T = pd.DataFrame(rows).set_index("method")
    T["p_BH"] = bh(T["p"].fillna(1).values)
    T = T.sort_values("mean_diff", ascending=False)
    T.to_csv(os.path.join(TAB, f"vs_gmm_{ds}.csv"))
    print(f"\n===== {ds}: method vs non-spatial GMM (paired across slides; ARI difference) =====")
    print(T.to_string())

print("\n===== ranking transfer DLPFC -> prostate =====")
base_methods = [m for m in means["DLPFC"].index if m in means["prostate"].index and not m.endswith("_refined")]
R = pd.DataFrame({"DLPFC": means["DLPFC"][base_methods], "prostate": means["prostate"][base_methods]})
R["rank_DLPFC"] = R["DLPFC"].rank(ascending=False).astype(int); R["rank_prostate"] = R["prostate"].rank(ascending=False).astype(int)
rho, p = spearmanr(R["DLPFC"], R["prostate"])
print(R.sort_values("rank_DLPFC").to_string()); print(f"Spearman rho = {rho:.3f}, p = {p:.3f}, n methods = {len(R)}")
R.to_csv(os.path.join(TAB, "rank_transfer.csv"))

print("\n===== spatial coherence of the pathologist labels themselves =====")
rows = []
for ds, pat in [("DLPFC", "Maynard"), ("prostate", "Erickson")]:
    for sid in bu.list_slides(pat):
        a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), backed="r")
        obs = a.obs.copy(); xy = np.asarray(a.obsm["spatial"]); a.file.close()
        lab = obs["label"].astype(object); lab = lab.where(~lab.astype(str).isin(bu.EXCLUDE_LABELS)).where(lab.notna())
        m = lab.notna().values
        vc = lab[m].value_counts(normalize=True)
        rows.append(dict(dataset=ds, slide=sid[-6:], n_scored=int(m.sum()), n_classes=int((lab[m].value_counts() >= bu.MIN_CLASS_SPOTS).sum()),
                         largest_class_frac=float(vc.iloc[0]), label_pas=pas(xy[m], lab[m].astype(str).values)))
G = pd.DataFrame(rows); G.to_csv(os.path.join(TAB, "ground_truth_coherence.csv"), index=False)
print(G.to_string(index=False)); print(G.groupby("dataset")[["label_pas", "largest_class_frac", "n_classes"]].mean().to_string())

print("\n===== smoothness (PAS) vs accuracy (ARI) across methods =====")
for ds in ["DLPFC", "prostate"]:
    d = ok[(ok["dataset"] == ds) & ~ok["method"].str.endswith("_refined")].groupby("method")[["pas", "ari"]].mean()
    rho, p = spearmanr(d["pas"], d["ari"])
    print(f"{ds}: Spearman rho(PAS, ARI) = {rho:.3f} (p = {p:.3f}, n = {len(d)})  [negative = smoother methods are more accurate]")

for ds in ["DLPFC", "prostate"]:
    print(f"\n===== {ds}: variability over seeds (SD of ARI within slide, mean over slides) =====")
    d = ok[ok["dataset"] == ds]
    sd = d.groupby(["method", "slide"])["ari"].agg(["std", "count"]).groupby("method").agg(mean_sd=("std", "mean"), runs=("count", "sum"))
    print(sd.sort_values("mean_sd").to_string()); sd.to_csv(os.path.join(TAB, f"seed_variability_{ds}.csv"))

# ---------- per slide: how fragmented are the pathologist labels, and how much do spatial methods gain over GMM? ----------
print("\n===== per slide: label fragmentation (PAS of pathologist labels) vs spatial-method gain over GMM =====")
SPATIAL = ["sedr", "graphst", "smooth", "stagate", "bayesspace", "spagcn", "banksy", "spaceflow", "drsc", "bass", "precast"]
rows = []
for ds in ["DLPFC", "prostate"]:
    per = ok[ok["dataset"] == ds].groupby(["method", "slide"])["ari"].mean().unstack()
    for sid in per.columns:
        sm = [m for m in SPATIAL if m in per.index and pd.notna(per.loc[m, sid])]
        rows.append(dict(dataset=ds, slide=sid[-6:], gain=float(per.loc[sm, sid].mean() - per.loc["gmm", sid]),
                         best_spatial_gain=float(per.loc[sm, sid].max() - per.loc["gmm", sid])))
A = pd.DataFrame(rows).merge(G[["dataset", "slide", "label_pas", "largest_class_frac", "n_classes"]], on=["dataset", "slide"])
A.to_csv(os.path.join(TAB, "gain_vs_label_fragmentation.csv"), index=False)
print(A.to_string(index=False))
rho, p = spearmanr(A["label_pas"], A["gain"]); print(f"all slides (n={len(A)}): Spearman rho(label PAS, mean spatial gain) = {rho:.3f}, p = {p:.4f}")
B = A[A["dataset"] == "prostate"]; rho, p = spearmanr(B["label_pas"], B["gain"])
print(f"prostate only (n={len(B)}): rho = {rho:.3f}, p = {p:.3f}")

# ---------- ccRCC: tertiary lymphoid structure detection ----------
d = ok[(ok["dataset"] == "ccRCC") & ok["tls_f1"].notna()] if "tls_f1" in ok else pd.DataFrame()
if not d.empty:
    print(f"\n===== ccRCC TLS: best-matching-cluster F1 (slides with >=20 TLS spots: {d['slide'].nunique()}) =====")
    byk = d.groupby(["method", "k"])["tls_f1"].mean().unstack(); byk.columns = [f"F1_K{c}" for c in byk.columns]
    per = d.groupby(["method", "slide"])["tls_f1"].mean().unstack()          # mean over K = 6 and 10
    per = per.loc[:, per.notna().all(0)]                                        # slides where every method finished
    summ = pd.DataFrame({"n_slides": per.shape[1], "mean_F1": per.mean(1), "median_F1": per.median(1)})
    rows = []
    for m in per.index:
        if m == "gmm":
            continue
        diff = (per.loc[m] - per.loc["gmm"]).values
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        rows.append(dict(method=m, diff_vs_gmm=diff.mean(), n_better=int((diff > 0).sum()), n_worse=int((diff < 0).sum()), p=p))
    T = pd.DataFrame(rows).set_index("method"); T["p_BH"] = bh(T["p"].values)
    TL = summ.join(byk).join(T).sort_values("mean_F1", ascending=False)
    pr = d.groupby("method")[["tls_precision", "tls_recall"]].mean()
    TL = TL.join(pr)
    TL.to_csv(os.path.join(TAB, "tls_ccrcc.csv")); per.to_csv(os.path.join(TAB, "tls_f1_per_slide.csv"))
    print(TL.to_string())
    base = d.drop_duplicates("slide")[["slide", "tls_prevalence", "tls_f1_all_positive", "tls_n_pos"]]
    print(f"reference: TLS spots per slide median {base['tls_n_pos'].median():.0f} (range {base['tls_n_pos'].min()}-{base['tls_n_pos'].max()}), "
          f"prevalence mean {base['tls_prevalence'].mean():.3f}; F1 of calling every spot TLS = {base['tls_f1_all_positive'].mean():.3f}")
print("\ntables in:", TAB)
