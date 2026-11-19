"""09_score.py
Score every finished run in results/runs against pathologist labels (when present) and with label-free
spatial-coherence metrics. Writes results/scores.csv and prints per-dataset summaries.

Metrics
- ARI, NMI, AMI, homogeneity, completeness: on scored spots only (labels 'Exclude'/missing are ignored)
- ARI_coarse / homogeneity_coarse (prostate only): labels merged to Tumor (GG1/GG2/GG4/GG4 Cribriform),
  Benign (Benign/Benign*), Stroma; all other classes unscored -> sensitivity analysis for label granularity
- CHAOS, PAS: spatial continuity of the predicted domains (lower = smoother), definitions as in SDMBench
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad
from scipy.spatial import cKDTree
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
                             homogeneity_completeness_v_measure)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

RUNS_DIR = os.path.join(bu.BASE, sys.argv[1]) if len(sys.argv) > 1 else bu.RUNS       # e.g. results/runs_ksens
OUT_CSV = os.path.join(bu.BASE, sys.argv[2]) if len(sys.argv) > 2 else os.path.join(bu.BASE, "results/scores.csv")

COARSE = {"GG1": "Tumor", "GG2": "Tumor", "GG4": "Tumor", "GG4 Cribriform": "Tumor",
          "Benign": "Benign", "Benign*": "Benign", "Stroma": "Stroma"}


def chaos(xy, lab):
    z = (xy - xy.mean(0)) / xy.std(0)
    tot = 0.0
    for c in np.unique(lab):
        p = z[lab == c]
        if len(p) <= 2:
            continue
        d, _ = cKDTree(p).query(p, k=2)
        tot += d[:, 1].sum()
    return float(tot / len(lab))


def pas(xy, lab, k=10):
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    diff = (lab[idx[:, 1:]] != lab[:, None]).sum(1)
    return float((diff > 6).mean())


def dataset_of(sid):
    if "Erickson" in sid: return "prostate"
    if "Maynard" in sid: return "DLPFC"
    if "GSE175540" in sid: return "ccRCC"
    return "other"


rows = []
for sdir in sorted(glob.glob(os.path.join(RUNS_DIR, "*"))):
    sid = os.path.basename(sdir)
    js = sorted(glob.glob(os.path.join(sdir, "*.json")))
    if not js:
        continue
    a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), backed="r")
    obs = a.obs.copy(); xy_all = pd.DataFrame(np.asarray(a.obsm["spatial"]), index=obs.index, columns=["x", "y"])
    a.file.close()
    for jf in js:
        r = json.load(open(jf))
        base = dict(slide=sid, dataset=dataset_of(sid), method=r["method"], k=r["k"], seed=r["seed"], status=r["status"],
                    sec=r.get("sec"), gpu_peak_mb=r.get("gpu_peak_mb"), n_found=r.get("n_found"))
        if r["status"] != "ok":
            rows.append(base); continue
        df = pd.read_csv(jf[:-5] + ".csv").set_index("spot")
        xy = xy_all.loc[df.index].values
        lab = None
        if "label" in obs:
            lab = obs.loc[df.index, "label"].astype(object)
            lab = lab.where(~lab.astype(str).isin(bu.EXCLUDE_LABELS)).where(lab.notna())
        for col, mname in [("cluster", r["method"]), ("cluster_refined", r["method"] + "_refined")]:
            if col not in df:
                continue
            pred = df[col].values
            row = dict(base, method=mname, n_found=int(len(np.unique(pred))), chaos=chaos(xy, pred), pas=pas(xy, pred))
            if lab is not None and lab.notna().sum() > 0:
                ok = lab.notna().values
                yt, yp = lab[ok].astype(str).values, pred[ok]
                h, c, v = homogeneity_completeness_v_measure(yt, yp)
                row.update(n_scored=int(ok.sum()), ari=adjusted_rand_score(yt, yp), nmi=normalized_mutual_info_score(yt, yp),
                           ami=adjusted_mutual_info_score(yt, yp), homogeneity=h, completeness=c)
                if base["dataset"] == "ccRCC":
                    # tertiary lymphoid structure (TLS) detection: best-matching cluster vs pathologist TLS spots
                    pos = (lab == "TLS").values; neg = (lab == "NO_TLS").values
                    if pos.sum() >= 20 and neg.sum() > 0:
                        best = (0.0, 0.0, 0.0)
                        for cl in np.unique(pred):
                            m = pred == cl
                            tp = int((m & pos).sum()); fp = int((m & neg).sum()); fn = int((~m & pos).sum())
                            if tp == 0:
                                continue
                            p_, r_ = tp / (tp + fp), tp / (tp + fn)
                            f = 2 * p_ * r_ / (p_ + r_)
                            if f > best[0]:
                                best = (f, p_, r_)
                        prev = pos.sum() / (pos.sum() + neg.sum())
                        row.update(tls_f1=best[0], tls_precision=best[1], tls_recall=best[2], tls_n_pos=int(pos.sum()),
                                   tls_prevalence=float(prev), tls_f1_all_positive=float(2 * prev / (1 + prev)))
                co = lab.map(COARSE)
                okc = co.notna().values
                if okc.sum() > 0 and co[okc].nunique() >= 2:
                    row.update(ari_coarse=adjusted_rand_score(co[okc].values, pred[okc]),
                               homogeneity_coarse=homogeneity_completeness_v_measure(co[okc].values, pred[okc])[0])
                    tum = (co == "Tumor").values
                    if tum.sum() >= 20:
                        # tumour detection (prostate): best-matching cluster vs all other scored spots
                        oth = ok & ~tum; f = 0.0
                        for cl in np.unique(pred):
                            m = pred == cl
                            tp = int((m & tum).sum())
                            if tp:
                                f = max(f, 2 * tp / (2 * tp + int((m & oth).sum()) + int((~m & tum).sum())))
                        row.update(tumor_f1=f, tumor_n_pos=int(tum.sum()))
            rows.append(row)

S = pd.DataFrame(rows)
out = OUT_CSV
S.to_csv(out, index=False)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format)
print("runs:", len(S), "| ok:", int((S["status"] == "ok").sum()), "| failed:", int((S["status"] != "ok").sum()))
if (S["status"] != "ok").any():
    print(S[S["status"] != "ok"].groupby(["dataset", "method"]).size().to_string())
ok = S[S["status"] == "ok"]
for ds in ["DLPFC", "prostate"]:
    d = ok[ok["dataset"] == ds]
    if d.empty or "ari" not in d:
        continue
    per = d.groupby(["method", "slide"])["ari"].mean().unstack()          # mean over seeds
    per.columns = [c[-6:] for c in per.columns]
    summ = pd.DataFrame({"n_slides": per.notna().sum(1), "mean_ARI": per.mean(1), "median_ARI": per.median(1)})
    extra = d.groupby("method")[[c for c in ["nmi", "homogeneity", "ari_coarse", "chaos", "pas", "sec"] if c in d]].mean()
    print(f"\n===== {ds}: ARI per slide (mean over seeds: {sorted(d['seed'].unique())}) =====")
    print(per.join(summ).sort_values("mean_ARI", ascending=False).to_string())
    print(f"--- {ds}: other metrics (mean) ---")
    print(extra.loc[summ.sort_values('mean_ARI', ascending=False).index].to_string())
d = ok[ok["dataset"] == "ccRCC"]
if not d.empty and "tls_f1" in d and d["tls_f1"].notna().any():
    d = d[d["tls_f1"].notna()]
    print(f"\n===== ccRCC: TLS detection, best-cluster F1 (slides scored: {d['slide'].nunique()}) =====")
    t = d.groupby(["method", "k"])["tls_f1"].mean().unstack()
    t["mean"] = t.mean(1)
    print(t.sort_values("mean", ascending=False).to_string())
    print("precision / recall of the best cluster (mean):")
    print(d.groupby("method")[["tls_precision", "tls_recall", "tls_prevalence", "tls_f1_all_positive"]].mean().to_string())
d = ok[~ok["dataset"].isin(["DLPFC", "prostate", "ccRCC"]) | (ok["dataset"].eq("ccRCC") & ok.get("tls_f1", pd.Series(index=ok.index, dtype=float)).isna())]
if not d.empty:
    print(f"\n===== slides without usable labels: spatial continuity (n slides = {d['slide'].nunique()}) =====")
    print(d.groupby("method")[["chaos", "pas", "sec"]].mean().sort_values("pas").to_string())
print("\nsaved:", out)
