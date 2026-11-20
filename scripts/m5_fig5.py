"""m5_fig5.py - main Figure 5: external test in 20 breast-cancer sections (panels a-o)."""
from mstyle import *
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment as lsa
import statsmodels.formula.api as smf

ALLM = list(METHODS)
SPM = [m for m in ALLM if m not in ("gmm", "leiden")]
DCOL = {"DLPFC": MUTED, "prostate": VIOLET, "breast": MAGENTA}
DNAME = {"DLPFC": "DLPFC", "prostate": "Prostate", "breast": "Breast"}
rng = np.random.default_rng(11)


def bshort(sid, tiny=False):
    if sid.startswith("GSE"):
        return sid.split("_")[1][-4:] if tiny else sid.split("_")[1]
    if sid.startswith("WU2021_"):
        s = sid.split("_")[1]
        return s[-4:] if tiny else f"Wu {s}"
    return ("A1" if "Block" in sid else "0609") if tiny else ("10x block A1" if "Block" in sid else "10x 06092021")


BCLS = [(("in situ", "dcis", "lcis"), MAGENTA, "in situ carcinoma"), (("surrounding",), ORANGE, "tumour margin"),
        (("invasive", "tumor", "tumour", "carcinoma", "spindle"), YELLOW, "invasive tumour"),
        (("normal", "healthy", "hyperplasia"), AQUA, "normal epithelium"), (("lymph", "til", "inflamm", "tls"), VIOLET, "lymphocytes"),
        (("adipose",), "#9ec5f4", "adipose"), (("necrosis",), INK2, "necrosis"), (("vascular", "endothel"), RED, "vessels"),
        (("nerve",), "#184f95", "nerve"), (("stroma", "fibros", "fibrous", "non tumor"), "#d6d4cb", "stroma")]


def bcol(lab):
    s = str(lab).lower()
    for keys, c, _ in BCLS:
        if any(k in s for k in keys):
            return c
    return MUTED


S = scores("results/scores_ext.csv"); S = S[S["ari"].notna() & S["method"].isin(ALLM)]
P = S.groupby(["slide", "method"])["ari"].mean().unstack()
GS = pd.read_csv(os.path.join(TAB, "ext_geometry_slide.csv")).set_index("slide")
P = P.loc[[s for s in P.index if s in GS.index]]
P["gain"] = P[[m for m in SPM if m in P]].mean(1) - P["gmm"]
B = GS[["boundary_frac", "n_classes", "n_scored"]].join(P)
S0 = scores(); S0 = S0[S0["dataset"].isin(["DLPFC", "prostate"]) & S0["method"].isin(ALLM)]
P0 = S0.groupby(["slide", "method"])["ari"].mean().unstack()
LPs = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")).set_index("slide")
R0 = LPs.loc[P0.index, ["dataset", "boundary_frac"]].join(P0)
R0["gain"] = R0[[m for m in SPM if m in R0]].mean(1) - R0["gmm"]
ALL = pd.concat([R0, B.assign(dataset="breast")])
order = P[ALLM].mean().sort_values(ascending=False).index.tolist()
Y = {m: len(order) - 1 - i for i, m in enumerate(order)}

fig = plt.figure(figsize=(FW, 9.5))
gs = GridSpec(5, 12, figure=fig, height_ratios=[0.85, 1.6, 1.3, 1.3, 1.3], hspace=0.78, wspace=2.2)

# a-c: annotations of the sections with the lowest, median and highest boundary share
srt = B.sort_values("boundary_frac")
rich = srt[(srt["n_classes"] >= 3) & (srt["n_scored"] >= 1500)]          # maps need several classes and a sizeable section
pick = [rich.index[0], rich.index[len(rich) // 2], srt.index[-1]]
seen = {}; maps0 = []
for j, (sid, lt) in enumerate(zip(pick, "abc")):
    lab, xy = load_section(sid)
    ax = fig.add_subplot(gs[0, j * 4:(j + 1) * 4]); maps0.append(ax)
    spatial(ax, xy.values, [bcol(v) if isinstance(v, str) else UNLABELLED for v in lab.values], fig)
    for v in lab.dropna().unique():
        seen[bcol(v)] = next(n for k, c, n in BCLS if c == bcol(v)) if bcol(v) != MUTED else "other"
    ptitle(ax, lt, f"{bshort(sid)}  ·  {B.loc[sid, 'boundary_frac']:.0%}", fontsize=FS)
top_align(maps0)
leg0 = [Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec=SURF, label=n) for c, n in seen.items()]

# d: ARI per section and method (columns ordered by boundary share)
ax = fig.add_subplot(gs[1, 0:8])
M = B.sort_values("boundary_frac")[order].T
heat(ax, M.values, SEQ[:11], 0, 0.8, fmt="{:.2f}", fs=4.8, xt=[f"{bshort(s, True)}  {b:.2f}" for s, b in
                                                               B.sort_values('boundary_frac')['boundary_frac'].items()],
     yt=[NAME.get(m, m) for m in order])
ax.tick_params(axis="x", labelsize=FMIN - 0.3, rotation=90); ax.tick_params(axis="y", labelsize=FMIN)
for lbl in ax.get_yticklabels():
    lbl.set_color(COL.get({v: k for k, v in NAME.items()}.get(lbl.get_text(), ""), INK))
ptitle(ax, "d", "ARI per section (columns: section, boundary share)")

# e: difference from GMM with bootstrap CI
ax = fig.add_subplot(gs[1, 9:12]); gridx(ax); zero(ax, "x")
for m in order:
    if m == "gmm":
        ax.text(0.01, Y[m], "GMM", fontsize=FMIN, color=INK2, va="center"); continue
    d = (P[m] - P["gmm"]).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    ax.plot(np.percentile(bs, [2.5, 97.5]), [Y[m]] * 2, color=COL[m], lw=1.1)
    ax.scatter(d.mean(), Y[m], s=16, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3)
    ax.text(1.04, Y[m], f"{int((d > 0).sum())}/{len(d)}", transform=ax.get_yaxis_transform(), fontsize=FMIN, color=INK2, va="center")
method_rows_m(ax, order, labels=False)
ax.set_xlabel("ΔARI vs GMM (95% CI)"); ptitle(ax, "e", "vs GMM")

# f: gain vs boundary share in all three data sets; g: ranks; h: class rule
ax = fig.add_subplot(gs[2, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
for ds in ["DLPFC", "prostate", "breast"]:
    g = ALL[ALL["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["gain"], s=18 if ds == "breast" else 13, color=DCOL[ds], edgecolor=SURF, lw=0.4, zorder=3)
rb = spearmanr(B["boundary_frac"], B["gain"]); ra = spearmanr(ALL["boundary_frac"], ALL["gain"])
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("ΔARI, spatial − GMM")
ax.text(0.97, 0.97, f"breast ρ = {rb[0]:.2f}\nall ρ = {ra[0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=FMIN, color=INK2)
ptitle(ax, "f", "Gain vs label grain")

ax = fig.add_subplot(gs[2, 4:8])
rk = pd.DataFrame({ds: ALL[ALL["dataset"] == ds][order].mean().rank(ascending=False) for ds in ["DLPFC", "prostate", "breast"]})
for m in order:
    hi = m in ("gmm", "smooth", "graphst", "bass")
    ax.plot([0, 1, 2], rk.loc[m].values, color=COL[m], lw=1.5 if hi else 0.7, alpha=1 if hi else 0.5, marker="o", ms=2.5, zorder=3 if hi else 2)
    if hi:
        ax.text(2.1, rk.loc[m, "breast"], NAME[m], fontsize=FMIN, va="center", color=COL[m])
ax.set_ylim(len(order) + 0.6, 0.4); ax.set_xlim(-0.2, 3.25); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["DLPFC", "Prost.", "Breast"])
ax.set_yticks([1, 5, 10]); ax.set_ylabel("Rank (1 = best)"); ax.spines["bottom"].set_visible(False)
ptitle(ax, "g", "Ranks move again")

RT = pd.read_csv(os.path.join(TAB, "ext_rule_test.csv"))
txt = open(os.path.join(TAB, "ext_rule_test.txt")).read()
coef = {}
for line in txt.split("test:")[0].splitlines():
    p = line.split()
    if p and p[0] in ("Intercept", "boundary_frac", "log10_n"):
        coef[p[0]] = float(p[1])
ax = fig.add_subplot(gs[2, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
ax.scatter(RT["boundary_frac"], RT["gain9"], s=np.clip(RT["n_spots"] / 25, 5, 40), color=MAGENTA, alpha=0.7, edgecolor=SURF, lw=0.4, zorder=3)
xx = np.linspace(0, 1, 50)
for n, ls in [(100, ":"), (500, "-"), (2000, "--")]:
    ax.plot(xx, coef["Intercept"] + coef["boundary_frac"] * xx + coef["log10_n"] * np.log10(n), color=INK, lw=1.0, ls=ls, label=f"{n:,}")
ax.set_ylim(-0.3, 0.5)
ax.legend(loc="upper right", fontsize=FMIN, handlelength=1.3, borderaxespad=0.1, title="prior rule, class size (spots)",
          title_fontsize=FMIN, ncol=3, columnspacing=0.6, handletextpad=0.3, frameon=True, framealpha=0.9, edgecolor="none")
ax.set_xlabel("Boundary share of the class"); ax.set_ylabel("ΔF1, spatial − GMM")
ptitle(ax, "h", f"{len(RT)} breast classes")

# i: predicted vs observed; j: rule accuracy by data set; k: PCA-smooth dose per breast section
ax = fig.add_subplot(gs[3, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax); zero(ax, "x")
ax.scatter(RT["pred"], RT["gain9"], s=14, color=MAGENTA, edgecolor=SURF, lw=0.4, zorder=3)
agree = (np.sign(RT["pred"]) == np.sign(RT["gain9"])).mean(); rr = spearmanr(RT["pred"], RT["gain9"])
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.text(0.03, 0.97, f"ρ = {rr[0]:.2f}, P = {rr[1]:.1g}", transform=ax.transAxes, va="top", fontsize=FMIN, color=INK2,
        bbox=dict(fc=SURF, ec="none", pad=0.8), zorder=5)
ax.set_xlabel("Predicted gain (prior rule)"); ax.set_ylabel("Observed gain")
ptitle(ax, "i", "Order transfers")

ax = fig.add_subplot(gs[3, 4:8]); gridy(ax)
CF = pd.read_csv(os.path.join(TAB, "class_f1.csv")); GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
W = CF[CF["dataset"].isin(["DLPFC", "prostate"])].pivot_table(index=["slide", "dataset", "cls"], columns="method", values="f1")
W = W.reset_index().merge(GC, on=["slide", "dataset", "cls"])
W["gain"] = W[[m for m in SP9 if m in W]].mean(1) - W["gmm"]; W["log10_n"] = np.log10(W["n_spots"])
W["pred"] = coef["Intercept"] + coef["boundary_frac"] * W["boundary_frac"] + coef["log10_n"] * W["log10_n"]
bars = [("DLPFC", W[W["dataset"] == "DLPFC"], "gain"), ("prostate", W[W["dataset"] == "prostate"], "gain"), ("breast", RT, "gain9")]
for i, (ds, g, oc) in enumerate(bars):
    a_ = float((np.sign(g[oc]) == np.sign(g["pred"])).mean())
    ax.bar(i, a_ * 100, color=DCOL[ds], width=0.6)
    ax.text(i, a_ * 100 + 2, f"{a_:.0%}\nn = {len(g)}", ha="center", va="bottom", fontsize=FMIN, color=INK2)
ax.axhline(50, color=MUTED, lw=0.7, ls="--")
ax.set_xticks(range(3)); ax.set_xticklabels(["DLPFC", "Prostate", "Breast\n(held out)"])
ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Sign of gain correct (%)")
ptitle(ax, "j", "Sign does not transfer")

ax = fig.add_subplot(gs[3, 8:12]); gridy(ax); zero(ax)
DE = pd.read_csv(os.path.join(bu.BASE, "results/dose/ext_dose.csv"))
DE = DE[(DE["variant"] == "all") & DE["ari"].notna() & DE["slide"].isin(GS.index)]
w = DE.groupby(["slide", "alpha"])["ari"].mean().unstack(); dl = w.sub(w[0.0], axis=0)
from matplotlib.colors import LinearSegmentedColormap
cmb = LinearSegmentedColormap.from_list("b", ["#f6c6d9", MAGENTA, "#7a1f45"])
bf = GS.loc[dl.index, "boundary_frac"]; nb_ = (bf - bf.min()) / (bf.max() - bf.min())
for s in dl.index:
    ax.plot(dl.columns, dl.loc[s].values, color=cmb(1 - nb_[s]), lw=0.8)
ax.plot(dl.columns, dl.mean().values, color=INK, lw=1.8, zorder=4)
ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
ax.set_xlabel("Neighbour weight α"); ax.set_ylabel("ΔARI vs α = 0")
ax.text(0.03, 0.03, "dark: low boundary share\nblack: mean", transform=ax.transAxes, fontsize=FMIN, color=INK2, va="bottom")
ptitle(ax, "k", "PCA-smooth dose")

# l-o: section with the highest boundary share: annotation and the domains of GMM, the best and the worst spatial method
sid = pick[-1]
lab, xy = load_section(sid)
cls = [c for c in lab.dropna().value_counts().index if (lab == c).sum() >= 20]
best = P.loc[sid, [m for m in SPM if m in P]].idxmax(); worst = P.loc[sid, [m for m in SPM if m in P]].idxmin()
maps4 = []
ax = fig.add_subplot(gs[4, 0:3]); maps4.append(ax)
spatial(ax, xy.values, [bcol(v) if isinstance(v, str) else UNLABELLED for v in lab.values], fig)
ptitle(ax, "l", f"{bshort(sid)}: annotation", fontsize=FS)
for j, (m, lt) in enumerate([("gmm", "m"), (best, "n"), (worst, "o")]):
    ax = fig.add_subplot(gs[4, 3 + 3 * j:6 + 3 * j]); maps4.append(ax)
    k = int(GS.loc[sid, "n_classes"])
    pr = run_labels(sid, m, k, 0, os.path.join(bu.BASE, "results/runs_ext")).reindex(xy.index)
    ok = lab.notna().values
    ucl = np.unique(pr.values)
    C = np.array([[np.sum((pr.values[ok] == u) & (lab.values[ok] == c)) for c in cls] for u in ucl])
    r, c = lsa(-C); mpc = {ucl[i]: bcol(cls[q]) for i, q in zip(r, c)}
    spatial(ax, xy.values, [mpc.get(v, MUTED) for v in pr.values], fig)
    ptitle(ax, lt, f"{NAME[m]}  ·  ARI {P.loc[sid, m]:.2f}", fontsize=FS)
top_align(maps4)
maps_legend(fig, maps4, leg0 + [Line2D([], [], marker="o", ls="", ms=5, mfc=UNLABELLED, mec=AXIS, mew=0.4, label="not scored")],
            gap=0.006, ncol=6)
top_legend(fig, [Line2D([], [], marker="o", ls="", ms=5, mfc=DCOL[d], mec=SURF, label={"DLPFC": "DLPFC", "prostate": "Prostate cancer",
                                                                                          "breast": "Breast cancer"}[d]) for d in DCOL]
           + cat_handles(), y=0.93, ncol=6, cs=0.9)
msave(fig, "Figure5")
print("M5_DONE", pick, best, worst)
