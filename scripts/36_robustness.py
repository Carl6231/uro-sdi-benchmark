"""36_robustness.py
Robustness and mechanism analyses that re-use the finished benchmark runs (no new clustering).
 1. Bootstrap 95% CIs (sections resampled, 10,000 draws) of the paired difference of each method vs GMM:
    ARI (DLPFC, prostate); coarse ARI and tumour best-cluster F1 (prostate); TLS F1 (ccRCC)
 2. Geometry of the reference labels: patches of identical labels on the Visium lattice (ring-1 adjacency);
    boundary fraction = share of labelled spots that touch a spot carrying another label
 3. Class-level best-cluster F1 for every annotated class (>= 20 spots): is the gain of spatial methods over GMM
    smaller for patchier classes? (Spearman with a section-level cluster bootstrap; mixed model with a random
    intercept per section, adjusted for class size)
 4. Majority filter: the same runs scored against reference labels smoothed by a 1- or 2-ring majority vote
Outputs in results/tables/: bootstrap_vs_gmm.csv, label_patches_slide.csv, label_patches_class.csv, class_f1.csv,
    class_gain_vs_patchiness.csv, class_gain_models.txt, majority_filter.csv, majority_filter_labels.csv
"""
import os, sys, glob, warnings
from collections import Counter
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

T = os.path.join(bu.BASE, "results/tables"); os.makedirs(T, exist_ok=True)
S = pd.read_csv(os.path.join(bu.BASE, "results/scores.csv")); S = S[S["status"] == "ok"]
SPATIAL9 = ["graphst", "stagate", "sedr", "spagcn", "spaceflow", "bayesspace", "banksy", "drsc", "smooth", "bass", "precast"]   # all spatial methods
rng = np.random.default_rng(2026)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200, "display.float_format", "{:.3f}".format)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1.0)
    return out


def boot_ci(x, B=10000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ---------------- 1. bootstrap CIs of the paired difference vs GMM ----------------
blocks = []
for ds, met in [("DLPFC", "ari"), ("prostate", "ari"), ("prostate", "ari_coarse"), ("prostate", "tumor_f1"), ("ccRCC", "tls_f1")]:
    if met not in S:
        continue
    d = S[(S["dataset"] == ds) & S[met].notna()]
    per = d.groupby(["method", "slide"])[met].mean().unstack().dropna(axis=1)      # mean over seeds (and K for ccRCC)
    if "gmm" not in per.index or per.shape[1] < 3:
        continue
    g = per.loc["gmm"]; rows = []
    for m in per.index.drop("gmm"):
        diff = (per.loc[m] - g).values
        lo, hi = boot_ci(diff)
        rows.append(dict(dataset=ds, metric=met, method=m, n_slides=len(diff), mean_method=per.loc[m].mean(),
                         mean_gmm=g.mean(), mean_diff=diff.mean(), ci95_lo=lo, ci95_hi=hi, n_better=int((diff > 0).sum()),
                         wilcoxon_p=wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0))
    b = pd.DataFrame(rows); b["q_bh"] = bh(b["wilcoxon_p"])
    blocks.append(b.sort_values("mean_diff", ascending=False))
BT = pd.concat(blocks, ignore_index=True)
BT.to_csv(os.path.join(T, "bootstrap_vs_gmm.csv"), index=False)
print("===== 1. difference vs GMM, mean over sections [bootstrap 95% CI] =====")
print(BT[["dataset", "metric", "method", "n_slides", "mean_method", "mean_gmm", "mean_diff", "ci95_lo", "ci95_hi",
          "n_better", "q_bh"]].to_string(index=False))


# ---------------- 2. geometry of the reference labels ----------------
def load_labels(sid):
    a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), backed="r")
    obs = a.obs.copy(); xy = np.asarray(a.obsm["spatial"], dtype=float); a.file.close()
    if "label" not in obs:
        return None, xy
    lab = obs["label"].astype(object)
    lab = lab.where(~lab.astype(str).isin(bu.EXCLUDE_LABELS)).where(lab.notna())
    return lab.map(lambda v: v if pd.isna(v) else str(v)), xy


def ring_edges(xy, factor):
    nb, _ = bu.lattice_neighbors(xy, factor)
    r = np.concatenate([np.full(len(v), i, dtype=int) for i, v in enumerate(nb)])
    return nb, r, np.concatenate(nb).astype(int)


def label_patches(v, r, c):
    """connected patches of identical scored labels (ring-1 lattice adjacency) and boundary spots"""
    s = pd.notna(v); n = len(v)
    same = s[r] & s[c] & (v[r] == v[c])
    G = sp.csr_matrix((np.ones(int(same.sum())), (r[same], c[same])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    psize = np.zeros(n)
    psize[s] = pd.Series(comp[s]).value_counts().reindex(comp[s]).values
    diff = s[r] & s[c] & (v[r] != v[c])
    bnd = np.zeros(n, dtype=bool); bnd[np.unique(r[diff])] = True
    return comp, psize, bnd, s


dsmap = S.groupby("slide")["dataset"].first()
geo_s, geo_c, cache = [], [], {}
for sid, ds in dsmap.items():
    if ds not in ("DLPFC", "prostate", "ccRCC"):
        continue
    lab, xy = load_labels(sid)
    if lab is None or lab.notna().sum() == 0:
        continue
    v = lab.values
    nb1, r1, c1 = ring_edges(xy, 1.1)
    comp, psize, bnd, s = label_patches(v, r1, c1)
    cache[sid] = (lab, xy, nb1, r1, c1, ds)
    vc = pd.Series(v[s]).value_counts(); classes = list(vc[vc >= bu.MIN_CLASS_SPOTS].index)
    geo_s.append(dict(slide=sid, dataset=ds, n_scored=int(s.sum()), n_classes=len(classes),
                      n_patches=int(len(np.unique(comp[s]))), spot_weighted_patch=float(psize[s].mean()),
                      frac_small_patch=float((psize[s] < 10).mean()), boundary_frac=float(bnd[s].mean())))
    for cl in classes:
        m = s & (v == cl)
        pc = pd.Series(comp[m]).value_counts()
        geo_c.append(dict(slide=sid, dataset=ds, cls=cl, n_spots=int(m.sum()), n_patches=int(len(pc)),
                          largest_patch_share=float(pc.max() / m.sum()), spot_weighted_patch=float(psize[m].mean()),
                          frac_small_patch=float((psize[m] < 10).mean()), boundary_frac=float(bnd[m].mean())))
GS, GC = pd.DataFrame(geo_s), pd.DataFrame(geo_c)
GS.to_csv(os.path.join(T, "label_patches_slide.csv"), index=False)
GC.to_csv(os.path.join(T, "label_patches_class.csv"), index=False)
print("\n===== 2. reference-label geometry (median over sections) =====")
print(GS.groupby("dataset")[["n_scored", "n_classes", "n_patches", "spot_weighted_patch", "frac_small_patch",
                             "boundary_frac"]].median().to_string())
print("per class (median over sections):")
print(GC.groupby(["dataset", "cls"])[["n_spots", "n_patches", "largest_patch_share", "spot_weighted_patch",
                                      "boundary_frac"]].median().to_string())


# ---------------- 3 + 4. class-level F1 and majority-filtered references (re-reading the run files) ----------------
def best_f1_by_class(yt, yp, classes):
    ct = pd.crosstab(yt, yp); col = ct.sum(0).values
    return {cl: float((2 * ct.loc[cl].values / (ct.loc[cl].values.sum() + col)).max()) for cl in classes if cl in ct.index}


def majority(v, nb):
    s = pd.notna(v); out = v.copy()
    for i in np.where(s)[0]:
        cnt = Counter([v[i]] + [v[j] for j in nb[i] if s[j]])
        top = max(cnt.values())
        if cnt[v[i]] < top:
            out[i] = sorted(x for x, cn in cnt.items() if cn == top)[0]
    return out


cf, mf, mfl = [], [], []
for sid, (lab, xy, nb1, r1, c1, ds) in cache.items():
    if ds not in ("DLPFC", "prostate"):
        continue
    v = lab.values
    classes = GC.loc[GC["slide"] == sid, "cls"].tolist()
    nb2, _, _ = ring_edges(xy, 2.1)
    levels = {0: v, 1: majority(v, nb1), 2: majority(v, nb2)}
    for lev, vv in levels.items():
        _, _, bnd, s = label_patches(vv, r1, c1)
        mfl.append(dict(slide=sid, dataset=ds, level=lev, boundary_frac=float(bnd[s].mean()),
                        changed=float(np.mean(vv[s] != v[s]))))
    labs = {lev: pd.Series(vv, index=lab.index) for lev, vv in levels.items()}
    for f in sorted(glob.glob(os.path.join(bu.RUNS, sid, "*.csv"))):
        meth, _, ss = os.path.basename(f)[:-4].rsplit("_", 2)
        df = pd.read_csv(f).set_index("spot")
        l0 = labs[0].loc[df.index]; ok = l0.notna().values
        for col, name in [("cluster", meth), ("cluster_refined", meth + "_refined")]:
            if col not in df:
                continue
            yp = df[col].values[ok]
            for cl, val in best_f1_by_class(l0.values[ok], yp, classes).items():
                cf.append(dict(slide=sid, dataset=ds, method=name, seed=int(ss[1:]), cls=cl, f1=val))
            for lev in (0, 1, 2):
                yt = labs[lev].loc[df.index].values[ok].astype(str)
                mf.append(dict(slide=sid, dataset=ds, method=name, seed=int(ss[1:]), level=lev, ari=adjusted_rand_score(yt, yp)))

CF = pd.DataFrame(cf).groupby(["slide", "dataset", "method", "cls"], as_index=False)["f1"].mean()
CF.to_csv(os.path.join(T, "class_f1.csv"), index=False)
W = CF.pivot_table(index=["slide", "dataset", "cls"], columns="method", values="f1").reset_index()
W = W.merge(GC, on=["slide", "dataset", "cls"], how="left")
sp9 = [m for m in SPATIAL9 if m in W]
W["spatial9"] = W[sp9].mean(1)
W["gain9"] = W["spatial9"] - W["gmm"]
W["log10_n"] = np.log10(W["n_spots"])
res = []
for m in sp9 + ["spatial9", "leiden"]:
    dlt = W[m] - W["gmm"]
    for scope in ["all", "DLPFC", "prostate"]:
        sel = np.ones(len(W), bool) if scope == "all" else (W["dataset"] == scope).values
        rho, p = spearmanr(W.loc[sel, "boundary_frac"], dlt[sel])
        res.append(dict(method=m, scope=scope, n_classes=int(sel.sum()), mean_gain=float(dlt[sel].mean()),
                        rho_vs_boundary=rho, p=p))
R = pd.DataFrame(res); R.to_csv(os.path.join(T, "class_gain_vs_patchiness.csv"), index=False)
sl = W["slide"].unique(); idx = {s_: np.where(W["slide"].values == s_)[0] for s_ in sl}
boot = []
for _ in range(2000):
    ii = np.concatenate([idx[s_] for s_ in rng.choice(sl, len(sl))])
    boot.append(spearmanr(W["boundary_frac"].values[ii], W["gain9"].values[ii])[0])
lo, hi = np.nanpercentile(boot, [2.5, 97.5])
print("\n===== 3. class-level best-cluster F1: gain of spatial methods over GMM vs class patchiness =====")
print(W.groupby("dataset")[["n_spots", "boundary_frac", "gmm", "spatial9", "gain9"]].mean().to_string())
print(R.to_string(index=False))
print(f"pooled 9-method gain vs boundary fraction: rho = {spearmanr(W['boundary_frac'], W['gain9'])[0]:.3f}, "
      f"section-bootstrap 95% CI [{lo:.3f}, {hi:.3f}], n classes = {len(W)}, sections = {len(sl)}")
lines = []
try:
    import statsmodels.formula.api as smf
    for form in ["gain9 ~ boundary_frac + log10_n", "gain9 ~ boundary_frac + log10_n + C(dataset)"]:
        fit = smf.mixedlm(form, W, groups=W["slide"]).fit(reml=True)
        lines.append(f"{form}   (random intercept per section)\n{fit.summary().tables[1].to_string()}\n")
except Exception as e:
    lines.append(f"mixed model not fitted: {e}")
open(os.path.join(T, "class_gain_models.txt"), "w").write("\n".join(lines))
print("\n".join(lines))
cols = ["dataset", "slide", "cls", "n_spots", "boundary_frac", "gmm", "spatial9", "gain9"]
print("classes with the largest loss / gain (mean of the 9 spatial methods minus GMM):")
print(W.sort_values("gain9")[cols].head(8).to_string(index=False))
print(W.sort_values("gain9")[cols].tail(8).to_string(index=False))

MF = pd.DataFrame(mf).groupby(["slide", "dataset", "method", "level"], as_index=False)["ari"].mean()
summ = []
for (ds, lev), d in MF.groupby(["dataset", "level"]):
    p = d.pivot(index="slide", columns="method", values="ari"); g = p["gmm"]
    for m in p.columns:
        dd = p[m] - g
        summ.append(dict(dataset=ds, level=lev, method=m, mean_ari=p[m].mean(), diff_vs_gmm=dd.mean(),
                         n_better=int((dd > 0).sum()), n_slides=len(p)))
MS = pd.DataFrame(summ)
MS["rank"] = MS.groupby(["dataset", "level"])["mean_ari"].rank(ascending=False, method="min").astype(int)
MS.to_csv(os.path.join(T, "majority_filter.csv"), index=False)
MFL = pd.DataFrame(mfl); MFL.to_csv(os.path.join(T, "majority_filter_labels.csv"), index=False)
print("\n===== 4. reference labels smoothed by a majority vote (level 0 = original, 1 = 1 ring, 2 = 2 rings) =====")
print(MFL.groupby(["dataset", "level"])[["boundary_frac", "changed"]].mean().to_string())
print(MS.pivot_table(index="method", columns=["dataset", "level"], values="mean_ari")
        .sort_values(("prostate", 0), ascending=False).to_string())
g9 = MS[MS["method"].isin(SPATIAL9)].groupby(["dataset", "level"]).agg(
    mean_gain=("diff_vs_gmm", "mean"), n_methods_above_gmm=("diff_vs_gmm", lambda x: int((x > 0).sum())))
print(g9.to_string())
print("ROBUSTNESS_DONE")
