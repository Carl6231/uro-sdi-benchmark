"""82_fig8_external.py - Figure 8: external test in breast cancer (panels a-o).
Annotated breast-cancer Visium sections from other patients and centres (STimage-1K4M: GSE213688, 10x Genomics).
The class-level rule (spatial gain ~ boundary share + log10 class size) is fitted on DLPFC + prostate and applied unchanged.
"""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment as lsa

ALLM = list(dict.fromkeys(METHODS + ["bass", "precast"]))
SPM = [m for m in ALLM if m not in ("gmm", "leiden")]
DCOL = {"DLPFC": MUTED, "prostate": VIOLET, "breast": MAGENTA}
DNAME = {"DLPFC": "DLPFC", "prostate": "Prostate cancer", "breast": "Breast cancer"}
rng = np.random.default_rng(11)


def bshort(sid, tiny=False):
    if sid.startswith("GSE"):
        return sid.split("_")[1][-4:] if tiny else sid.split("_")[1]
    if sid.startswith("WU2021_"):
        s = sid.split("_")[1]
        return s[-4:] if tiny else f"Wu {s}"
    return ("A1" if "Block" in sid else "0609") if tiny else ("10x block A1" if "Block" in sid else "10x 06092021")


def bcol(lab):
    s = str(lab).lower()
    for keys, c in [(("in situ", "dcis", "lcis"), MAGENTA), (("surrounding",), ORANGE), (("invasive", "tumor", "tumour", "carcinoma", "spindle"), YELLOW),
                    (("normal", "healthy", "hyperplasia"), AQUA), (("lymph", "til", "inflamm", "tls"), VIOLET), (("adipose",), "#9ec5f4"),
                    (("necrosis",), INK2), (("vascular", "endothel"), RED), (("nerve",), "#184f95"),
                    (("stroma", "fibros", "fibrous", "non tumor"), "#d6d4cb")]:
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

fig = plt.figure(figsize=(7.4, 10.6))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.3, 1.45, 1.3, 1.3, 1.3], hspace=0.8, wspace=2.4)

# a-c: annotations of three sections spanning the range of boundary share
srt = B.sort_values("boundary_frac")
pick = [srt.index[0], srt.index[len(srt) // 2], srt.index[-1]]
map_axes = []; seen = {}
for j, (sid, lt) in enumerate(zip(pick, "abc")):
    lab, xy = load_section(sid)
    ax = fig.add_subplot(gs[0, j * 4:(j + 1) * 4]); map_axes.append(ax)
    spatial(ax, xy.values, [bcol(v) if isinstance(v, str) else UNLABELLED for v in lab.values], fig)
    for v in lab.dropna().unique():
        seen.setdefault(bcol(v), v)
    nm = bshort(sid)
    ax.set_title(f"{lt}   {nm}: {B.loc[sid, 'boundary_frac']:.0%} boundary", loc="left", fontsize=6.2)
legend_below(fig, map_axes, [Line2D([], [], marker="o", ls="", ms=4.5, mfc=c, mec=SURF, label=str(v)[:22]) for c, v in seen.items()]
             + [Line2D([], [], marker="o", ls="", ms=4.5, mfc=UNLABELLED, mec=AXIS, mew=0.4, label="not scored")], gap=0.004, fs=5.0,
             ncol=5)

# d: grain of the annotations in the three datasets
ax = fig.add_subplot(gs[1, 0:3]); gridx(ax)
for yy, ds in enumerate(["breast", "prostate", "DLPFC"]):
    v = ALL.loc[ALL["dataset"] == ds, "boundary_frac"].values
    ax.scatter(v, yy + rng.uniform(-0.15, 0.15, len(v)), s=9, color=DCOL[ds], alpha=0.75, lw=0)
    ax.plot([np.median(v)] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=1.0)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels([f"Breast ({(ALL.dataset == 'breast').sum()})", f"Prostate ({(ALL.dataset == 'prostate').sum()})",
                                              f"DLPFC ({(ALL.dataset == 'DLPFC').sum()})"], fontsize=5.0)
ax.set_ylim(-0.5, 2.5); ax.set_xlabel("Boundary share of the labels"); ax.set_title("d   Label grain", loc="left")

# e: ARI per method and section
ax = fig.add_subplot(gs[1, 4:12])
order = P[ALLM if all(m in P for m in ALLM) else [m for m in ALLM if m in P]].mean().sort_values(ascending=False).index.tolist()
M = B.sort_values("boundary_frac")[order].T
heat(ax, M.values, SEQ[:11], 0, 0.8, fmt="{:.2f}", fs=3.4, xt=[f"{bshort(s, True)}\n{b:.2f}"
                                                              for s, b in B.sort_values('boundary_frac')['boundary_frac'].items()],
     yt=[NAME.get(m, m) for m in order])
ax.tick_params(axis="x", labelsize=4.4); ax.tick_params(axis="y", labelsize=4.8)
ax.set_title("e   ARI per section (columns: section, boundary share)", loc="left")

# f: gain vs boundary share in all three datasets
ax = fig.add_subplot(gs[2, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
for ds in ["DLPFC", "prostate", "breast"]:
    g = ALL[ALL["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["gain"], s=13 if ds == "breast" else 10, color=DCOL[ds], edgecolor=SURF, lw=0.4, zorder=3, label=DNAME[ds])
rb = spearmanr(B["boundary_frac"], B["gain"]); ra = spearmanr(ALL["boundary_frac"], ALL["gain"])
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel(f"ARI, spatial (mean of {len(SPM)}) − GMM")
ax.legend(loc="lower left", fontsize=4.6, handletextpad=0.1)
ax.set_title(f"f   Breast ρ = {rb[0]:.2f}; all ρ = {ra[0]:.2f}", loc="left")

# g: difference from GMM per method (bootstrap over breast sections)
ax = fig.add_subplot(gs[2, 5:9]); gridx(ax); zero(ax, "x")
om = [m for m in order if m != "gmm"]
for i, m in enumerate(om):
    d = (P[m] - P["gmm"]).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    y = len(om) - 1 - i
    ax.plot(np.percentile(bs, [2.5, 97.5]), [y, y], color=COL.get(m, BLUE), lw=1.0)
    ax.scatter(d.mean(), y, s=13, color=COL.get(m, BLUE), edgecolor=SURF, lw=0.5, zorder=3)
    ax.text(1.02, y, f"{int((d > 0).sum())}/{len(d)}", transform=ax.get_yaxis_transform(), fontsize=4.6, color=INK2, va="center")
ax.set_yticks(range(len(om))); ax.set_yticklabels([NAME.get(m, m) for m in om][::-1], fontsize=4.8); ax.set_ylim(-0.6, len(om) - 0.4)
ax.set_xlabel("ΔARI vs GMM, breast (95% CI)"); ax.set_title("g   Difference from GMM", loc="left")

# h: rank of every method in the three datasets
ax = fig.add_subplot(gs[2, 10:12])
rk = pd.DataFrame({ds: ALL[ALL["dataset"] == ds][order].mean().rank(ascending=False) for ds in ["DLPFC", "prostate", "breast"]})
for m in order:
    ax.plot([0, 1, 2], rk.loc[m].values, color=COL.get(m, BLUE), lw=1.2 if m in ("gmm", "smooth") else 0.6,
            alpha=1 if m in ("gmm", "smooth") else 0.6, marker="o", ms=1.8)
for m in order:
    ax.text(2.15, rk.loc[m, "breast"], NAME.get(m, m), fontsize=4.3, va="center", color=INK2)
ax.set_ylim(len(order) + 0.6, 0.4); ax.set_xlim(-0.2, 2.1); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["DLPFC", "Prost.", "Breast"], fontsize=4.8)
ax.set_yticks([1, 5, 10]); ax.set_ylabel("Rank (1 = best)"); [ax.spines[s_].set_visible(False) for s_ in ["bottom"]]
ax.set_title("h   Ranks", loc="left")

# i, j: class-level rule, fitted on DLPFC + prostate, tested on breast
RT = pd.read_csv(os.path.join(TAB, "ext_rule_test.csv"))
txt = open(os.path.join(TAB, "ext_rule_test.txt")).read()
ax = fig.add_subplot(gs[3, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
ax.scatter(RT["boundary_frac"], RT["gain9"], s=np.clip(RT["n_spots"] / 25, 4, 30), color=MAGENTA, alpha=0.75, edgecolor=SURF, lw=0.4, zorder=3)
coef = {}
for line in txt.split("test:")[0].splitlines():
    p = line.split()
    if p and p[0] in ("Intercept", "boundary_frac", "log10_n"):
        coef[p[0]] = float(p[1])
xx = np.linspace(0, 1, 50)
for n, ls in [(100, ":"), (500, "-"), (2000, "--")]:
    ax.plot(xx, coef["Intercept"] + coef["boundary_frac"] * xx + coef["log10_n"] * np.log10(n), color=INK, lw=0.8, ls=ls,
            label=f"rule, {n} spots")
ax.legend(loc="lower left", fontsize=4.5, handlelength=1.6)
ax.set_xlabel("Boundary share of the class"); ax.set_ylabel(f"ΔF1, spatial (mean) − GMM")
ax.set_title("i   Classes vs the prior rule", loc="left")
ax = fig.add_subplot(gs[3, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax); zero(ax, "x")
ax.scatter(RT["pred"], RT["gain9"], s=10, color=MAGENTA, edgecolor=SURF, lw=0.4, zorder=3)
agree = (np.sign(RT["pred"]) == np.sign(RT["gain9"])).mean(); rr = spearmanr(RT["pred"], RT["gain9"])
ax.set_xlabel("Predicted gain (DLPFC + prostate rule)"); ax.set_ylabel("Observed gain, breast")
ax.set_title(f"j   Out of sample: ρ = {rr[0]:.2f}, sign {agree:.0%}", loc="left")

# k, l: PCA-smooth dose in breast
DE = pd.read_csv(os.path.join(bu.BASE, "results/dose/ext_dose.csv"))
DE = DE[(DE["variant"] == "all") & DE["ari"].notna()]
w = DE.groupby(["slide", "alpha"])["ari"].mean().unstack()
dl = w.sub(w[0.0], axis=0)
ax = fig.add_subplot(gs[3, 8:12]); gridy(ax); zero(ax)
from matplotlib.colors import LinearSegmentedColormap
cmb = LinearSegmentedColormap.from_list("b", SEQ[3:])
bf = GS.loc[dl.index, "boundary_frac"]; nb_ = (bf - bf.min()) / (bf.max() - bf.min())
for s in dl.index:
    ax.plot(dl.columns, dl.loc[s].values, color=cmb(1 - nb_[s]), lw=0.8)
ax.set_xlabel("Neighbour weight α (1 ring)"); ax.set_ylabel("ΔARI vs α = 0")
ax.set_title("k   Smoothing dose (dark: smooth labels)", loc="left")

# l: best alpha vs grain, all datasets
ax = fig.add_subplot(gs[4, 0:3]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
bA = w.idxmax(1)
D0 = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
b0 = D0[D0["ring"].isin([0, 1])].groupby(["slide", "alpha"])["ari"].mean().unstack().idxmax(1)
pts = []
for ds in ["DLPFC", "prostate"]:
    ss = [s for s in b0.index if s in LPs.index and LPs.loc[s, "dataset"] == ds]
    ax.scatter(LPs.loc[ss, "boundary_frac"], b0.loc[ss] + rng.uniform(-0.015, 0.015, len(ss)), s=10, color=DCOL[ds], edgecolor=SURF, lw=0.3)
    pts += list(zip(LPs.loc[ss, "boundary_frac"], b0.loc[ss]))
ax.scatter(GS.loc[bA.index, "boundary_frac"], bA + rng.uniform(-0.015, 0.015, len(bA)), s=13, color=MAGENTA, edgecolor=SURF, lw=0.3)
pts += list(zip(GS.loc[bA.index, "boundary_frac"], bA))
r_ = spearmanr([p[0] for p in pts], [p[1] for p in pts])
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("Best α (1 ring)")
ax.set_title(f"l   Best smoothing (ρ = {r_[0]:.2f})", loc="left")

# m-o: one high-boundary breast section: annotation-matched domains of GMM and two spatial methods
sid = pick[-1]
lab, xy = load_section(sid)
cls = [c for c in lab.dropna().value_counts().index if (lab == c).sum() >= 20]
best = P.loc[sid, [m for m in SPM if m in P]].idxmax(); worst = P.loc[sid, [m for m in SPM if m in P]].idxmin()
for j, (m, lt) in enumerate([("gmm", "m"), (best, "n"), (worst, "o")]):
    ax = fig.add_subplot(gs[4, 3 + 3 * j:6 + 3 * j])
    k = int(GS.loc[sid, "n_classes"])
    pr = run_labels(sid, m, k, 0, os.path.join(bu.BASE, "results/runs_ext")).reindex(xy.index)
    ok = lab.notna().values
    ucl = np.unique(pr.values)
    C = np.array([[np.sum((pr.values[ok] == u) & (lab.values[ok] == c)) for c in cls] for u in ucl])
    r, c = lsa(-C); mpc = {ucl[i]: bcol(cls[q]) for i, q in zip(r, c)}
    spatial(ax, xy.values, [mpc.get(v, MUTED) for v in pr.values], fig)
    ax.set_title(f"{lt}   {NAME.get(m, m)}: ARI {P.loc[sid, m]:.2f}", loc="left", fontsize=6.0, fontweight="normal")
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.5, mfc=DCOL[d], mec=SURF, label=DNAME[d]) for d in DCOL] + category_handles()[:3],
           loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=6, fontsize=5.2, handletextpad=0.2, columnspacing=0.9)
B.to_csv(os.path.join(TAB, "ext_sections_summary.csv"))
save(fig, "Fig8_external_breast")
print("FIG8_DONE")
