"""72_followup_tables.py - tables for the follow-up analyses (simulation, external breast-cancer data, pooled rule).

A. Simulation: geometry of the simulated labels, ARI per method, spatial gain (mean of all spatial methods - GMM)
   and the boundary share at which each method stops beating GMM (crossover, bootstrap CI over replicates).
B. External breast cancer: geometry of the pathologist labels (sections and classes), ARI per method (mean of seeds),
   class-level best-cluster F1, and an out-of-sample test of the class-level model fitted on DLPFC + prostate
   (gain9 ~ boundary share + log10 class size, random intercept per section).
C. Pooled section-level data (DLPFC, prostate, breast, simulation): boundary share vs spatial gain.
Outputs in results/tables/: sim_geometry.csv, sim_summary.csv, sim_crossover_linear.csv, ext_geometry_slide.csv,
ext_geometry_class.csv, ext_ari.csv, ext_class_f1.csv, ext_rule_test.csv, ext_rule_test.txt, pooled_sections.csv
"""
import os, sys, glob, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu
import geom_utils as gu

T = os.path.join(bu.BASE, "results/tables")
SP9 = ["graphst", "stagate", "sedr", "spagcn", "spaceflow", "bayesspace", "banksy", "drsc", "smooth", "bass", "precast"]  # all spatial methods
ALL11 = SP9 + ["gmm", "leiden"]
rng = np.random.default_rng(7)


def gain_table(S, metric="ari"):
    """section x method mean metric (over seeds), plus gain9 = mean of the spatial methods - GMM (name kept from v1)"""
    P = S[S["method"].isin(ALL11)].groupby(["slide", "method"])[metric].mean().unstack()
    P["gain9"] = P[[m for m in SP9 if m in P]].mean(1) - P["gmm"]
    return P


def crossover(x, y):
    """x where the linear fit of y on x crosses 0 (nan if slope >= 0)"""
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan
    b, a = np.polyfit(x, y, 1)
    return -a / b if b < 0 else np.nan


out = {}
# ---------------- A. simulation ----------------
fs = os.path.join(bu.BASE, "results/scores_sim.csv")
if os.path.exists(fs):
    S = pd.read_csv(fs); S = S[S["status"] == "ok"]
    meta = pd.read_csv(os.path.join(T, "sim_sections.csv")).set_index("slide")
    geo = [gu.geometry(s, "sim")[0] for s in meta.index]
    G = pd.DataFrame([g for g in geo if g]).set_index("slide").join(meta[["source", "ell", "rep"]])
    P = gain_table(S)
    G = G.join(P)
    G.to_csv(os.path.join(T, "sim_geometry.csv"))
    summ = G.groupby(["source", "ell"])[["boundary_frac"] + [m for m in ALL11 + ["gain9"] if m in G]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(T, "sim_summary.csv"))
    rows = []
    for src, g in G.groupby("source"):
        for m in [x for x in SP9 + ["gain9"] if x in G]:
            y = (g[m] - g["gmm"]) if m != "gain9" else g["gain9"]
            est = crossover(g["boundary_frac"].values, y.values)
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(g), len(g))
                bs.append(crossover(g["boundary_frac"].values[i], y.values[i]))
            bs = np.array(bs, float)
            rows.append(dict(source=src, method=m, crossover=est, ci_lo=np.nanpercentile(bs, 2.5) if np.isfinite(bs).any() else np.nan,
                             ci_hi=np.nanpercentile(bs, 97.5) if np.isfinite(bs).any() else np.nan,
                             frac_valid=float(np.isfinite(bs).mean()), rho=spearmanr(g["boundary_frac"], y)[0],
                             p=spearmanr(g["boundary_frac"], y)[1]))
    # sensitivity version (linear fit over all replicates); the figure's interpolated version is sim_crossover.csv (81_fig7_simulation.py)
    pd.DataFrame(rows).to_csv(os.path.join(T, "sim_crossover_linear.csv"), index=False)
    out["sim"] = G
    print("A. simulation:", len(G), "sections\n", pd.DataFrame(rows).round(3).to_string(index=False), flush=True)

# ---------------- B. external breast cancer ----------------
fe = os.path.join(bu.BASE, "results/scores_ext.csv")
if os.path.exists(fe):
    S = pd.read_csv(fe); S = S[(S["status"] == "ok") & S["ari"].notna()]
    slides = sorted(S["slide"].unique())
    gs, gc = [], []
    for s in slides:
        r, c = gu.geometry(s, "breast")
        if r:
            gs.append(r); gc.extend(c)
    GS, GC = pd.DataFrame(gs).set_index("slide"), pd.DataFrame(gc)
    P = gain_table(S)
    GS = GS.join(P); GS.to_csv(os.path.join(T, "ext_geometry_slide.csv")); GC.to_csv(os.path.join(T, "ext_geometry_class.csv"), index=False)
    S.groupby(["slide", "method"])[["ari", "nmi", "pas"]].mean().reset_index().to_csv(os.path.join(T, "ext_ari.csv"), index=False)
    # class-level best-cluster F1 from the run files (mean over seeds)
    cf = []
    for s in slides:
        lab, xy, names = gu.load_labels(s)
        lab.index = names.astype(str)
        classes = GC.loc[GC["slide"] == s, "cls"].tolist()
        for jf in glob.glob(os.path.join(bu.BASE, "results/runs_ext", s, "*.json")):
            r = json.load(open(jf))
            if r.get("status") != "ok" or r["method"] not in ALL11:
                continue
            df = pd.read_csv(jf[:-5] + ".csv").set_index("spot")
            yt = lab.reindex(df.index.astype(str))
            ok = yt.notna().values
            for cl, f in gu.best_f1_by_class(yt[ok].values, df["cluster"].values[ok], classes).items():
                cf.append(dict(slide=s, dataset="breast", method=r["method"], seed=r["seed"], cls=cl, f1=f))
    CF = pd.DataFrame(cf).groupby(["slide", "dataset", "method", "cls"])["f1"].mean().reset_index()
    CF.to_csv(os.path.join(T, "ext_class_f1.csv"), index=False)

    def class_gain(cf, gc):
        w = cf.pivot_table(index=["slide", "cls"], columns="method", values="f1")
        g = (w[[m for m in SP9 if m in w]].mean(1) - w["gmm"]).rename("gain9").reset_index()
        g = g.merge(gc[["slide", "cls", "n_spots", "boundary_frac"]], on=["slide", "cls"])
        g["log10_n"] = np.log10(g["n_spots"])
        return g

    import statsmodels.formula.api as smf
    train = class_gain(pd.read_csv(os.path.join(T, "class_f1.csv")).query("dataset in ['DLPFC', 'prostate']"),
                       pd.read_csv(os.path.join(T, "label_patches_class.csv")))
    test = class_gain(CF, GC)
    fit = smf.mixedlm("gain9 ~ boundary_frac + log10_n", train, groups=train["slide"]).fit()
    test["pred"] = fit.predict(test)
    test.to_csv(os.path.join(T, "ext_rule_test.csv"), index=False)
    agree = float((np.sign(test["pred"]) == np.sign(test["gain9"])).mean())
    rho = spearmanr(test["pred"], test["gain9"])
    rho_b = spearmanr(test["boundary_frac"], test["gain9"])
    fit_ext = smf.mixedlm("gain9 ~ boundary_frac + log10_n", test, groups=test["slide"]).fit()
    txt = (f"training: DLPFC + prostate classes n = {len(train)}\n{fit.summary().tables[1].to_string()}\n\n"
           f"test: breast classes n = {len(test)} in {test['slide'].nunique()} sections\n"
           f"sign agreement (predicted vs observed gain): {agree:.3f}\n"
           f"Spearman predicted vs observed: rho = {rho[0]:.3f}, P = {rho[1]:.2g}\n"
           f"Spearman boundary share vs observed gain: rho = {rho_b[0]:.3f}, P = {rho_b[1]:.2g}\n\n"
           f"model refitted on breast only:\n{fit_ext.summary().tables[1].to_string()}\n")
    open(os.path.join(T, "ext_rule_test.txt"), "w").write(txt)
    out["ext"] = GS
    print("B. external:", len(GS), "sections\n", txt, flush=True)

# ---------------- C. pooled section level ----------------
S0 = pd.read_csv(os.path.join(bu.BASE, "results/scores.csv")); S0 = S0[(S0["status"] == "ok") & S0["dataset"].isin(["DLPFC", "prostate"])]
P0 = gain_table(S0).join(pd.read_csv(os.path.join(T, "label_patches_slide.csv")).set_index("slide")[["dataset", "boundary_frac"]])
parts = [P0[["dataset", "boundary_frac", "gain9", "gmm"]]]
if "ext" in out:
    parts.append(out["ext"].assign(dataset="breast")[["dataset", "boundary_frac", "gain9", "gmm"]])
if "sim" in out:
    parts.append(out["sim"].assign(dataset="sim_" + out["sim"]["source"])[["dataset", "boundary_frac", "gain9", "gmm"]])
C = pd.concat(parts)
C.to_csv(os.path.join(T, "pooled_sections.csv"))
real = C[~C["dataset"].str.startswith("sim")]
print("C. pooled sections:", C["dataset"].value_counts().to_dict())
print("real sections: Spearman boundary vs gain9:", spearmanr(real["boundary_frac"], real["gain9"]),
      "crossover:", round(crossover(real["boundary_frac"].values, real["gain9"].values), 3))
print("FOLLOWUP_TABLES_DONE")
