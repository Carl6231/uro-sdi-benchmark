"""57_fig7_robustness.py - Figure 7: robustness of the conclusions (K, sections, seeds, metrics, controls) (panels a-o)."""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

S = scores(); main = S[S["method"].isin(METHODS)]
KP = pd.read_csv(os.path.join(TAB, "ksens_per_slide.csv")); KP = KP[KP["method"].isin(METHODS)]
FK = pd.read_csv(os.path.join(TAB, "fixedk_pas.csv"), index_col=0); LF = pd.read_csv(os.path.join(TAB, "label_free_diagnostic.csv"), index_col=0)
MS = pd.read_csv(os.path.join(TAB, "majority_filter.csv")); SO = pd.read_csv(os.path.join(TAB, "section_overview.csv"), index_col=0)
rng = np.random.default_rng(0)
DK = sorted(KP["dk"].unique())


def kline(ax, met, lt, tt, yl, gpos):
    """Lines of (method - GMM) across K; the bold blue line is GraphST (labelled at gpos), the bold green line PCA-smooth."""
    w = KP.pivot_table(index=["slide", "dk"], columns="method", values=met)
    dif = w.sub(w["gmm"], axis=0).groupby(level="dk").mean()
    for m in [x for x in METHODS if x != "gmm" and x in dif]:
        lw = 1.3 if m in ("graphst", "smooth") else 0.7
        ax.plot(dif.index, dif[m], color=COL[m], lw=lw, marker="o", ms=2, alpha=0.95 if lw > 1 else 0.55)
    zero(ax); gridy(ax)
    dk, side = gpos
    off, ha, va = ((-3, 0), "right", "center") if side == "left" else ((0, 4), "center", "bottom")
    ax.annotate(NAME["graphst"], (dk, dif.loc[dk, "graphst"]), xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=4.9, color=INK2)
    ax.set_xticks(DK); ax.set_xticklabels([f"{d:+d}" if d else "ref" for d in DK]); ax.set_xlim(DK[0] - 0.25, DK[-1] + 0.25)
    ax.set_xlabel("K − annotated classes"); ax.set_ylabel(yl); ax.set_title(f"{lt}   {tt}", loc="left")


fig = plt.figure(figsize=(7.4, 10.2))
gs = GridSpec(5, 12, figure=fig, hspace=0.85, wspace=2.4)

# a: ΔARI vs GMM across K; b: absolute ARI; c: best K per method
ax = fig.add_subplot(gs[0, 0:4]); kline(ax, "ari", "a", "Prostate: ΔARI vs GMM", "ΔARI vs GMM, same K", (-1, "left"))
ax = fig.add_subplot(gs[0, 4:8]); gridy(ax)
mean_k = KP.groupby(["method", "dk"])["ari"].mean().unstack()
for m in [x for x in METHODS if x in mean_k.index]:
    ax.plot(DK, mean_k.loc[m, DK], color=COL[m], lw=2.0 if m == "gmm" else 0.7, marker="o", ms=2, alpha=1 if m == "gmm" else 0.55)
ax.text(DK[-1] + 0.1, mean_k.loc["gmm", DK[-1]], "GMM", fontsize=5, va="center", color=INK2)
ax.set_xticks(DK); ax.set_xticklabels([f"{d:+d}" if d else "ref" for d in DK]); ax.set_xlim(DK[0] - 0.2, DK[-1] + 0.8)
ax.set_xlabel("K − annotated classes"); ax.set_ylabel("Mean ARI (sections available)")
ax.set_title("b   Absolute ARI across K", loc="left")
ax = fig.add_subplot(gs[0, 8:12]); gridx(ax)
best = KP.groupby(["slide", "method"])["ari"].max().unstack().mean().sort_values()
for i, (m, v) in enumerate(best.items()):
    ax.barh(i, v, color=COL[m], height=0.65)
    ax.text(v + 0.006, i, f"{NAME[m]} {v:.2f}", va="center", fontsize=4.9, color=INK2)
ax.set_yticks([]); ax.set_xlim(0, best.max() * 1.55); ax.set_ylim(-0.6, len(best) - 0.4)
ax.set_xlabel("Mean ARI at each method's best K"); ax.set_title("c   Best K for every method", loc="left")

# d, e: coarse ARI and tumour F1 across K; f: GMM rank table
ax = fig.add_subplot(gs[1, 0:4]); kline(ax, "ari_coarse", "d", "Coarse ARI across K", "Δcoarse ARI vs GMM", (-1, "left"))
ax = fig.add_subplot(gs[1, 4:8]); kline(ax, "tumor_f1", "e", "Tumour F1 across K", "ΔF1 vs GMM", (1, "above"))
ax = fig.add_subplot(gs[1, 8:12])
rk = []
for met in ["ari", "ari_coarse", "tumor_f1"]:
    m_ = KP.groupby(["dk", "method"])[met].mean().unstack()
    rk.append(m_.rank(axis=1, ascending=False, method="min")["gmm"].values)
rk = np.array(rk)
heat(ax, 12 - rk, SEQ[2:], 1, 11, text=False, xt=[f"{d:+d}" if d else "ref" for d in DK], yt=["ARI", "coarse", "tumour"])
for i in range(rk.shape[0]):
    for j in range(rk.shape[1]):
        ax.text(j, i, f"{int(rk[i, j])}", ha="center", va="center", fontsize=6, color=SURF if rk[i, j] <= 5 else INK)
ax.set_xlabel("K − annotated classes"); ax.set_title(f"f   GMM rank of {KP['method'].nunique()} (1 = best)", loc="left")

# g: leave-one-section-out; h: per-seed ranking
ax = fig.add_subplot(gs[2, 0:4]); gridx(ax); zero(ax, "x")
for yy, ds in enumerate(["prostate", "DLPFC"]):
    P = main[main["dataset"] == ds].groupby(["method", "slide"])["ari"].mean().unstack()
    vals, ranks = [], []
    for s in P.columns:
        mm = P.drop(columns=s).mean(1)
        vals.append(mm[[x for x in SP9 if x in mm]].max() - mm["gmm"]); ranks.append(int(mm.rank(ascending=False)["gmm"]))
    ax.scatter(vals, yy + rng.uniform(-0.15, 0.15, len(vals)), s=12, color=DSCOL[ds], edgecolor=SURF, lw=0.4, zorder=3)
    rtxt = f"GMM rank {min(ranks)} in all {len(ranks)}" if min(ranks) == max(ranks) else f"GMM rank {min(ranks)}–{max(ranks)}"
    ax.text(0.99, yy + 0.32, rtxt, transform=ax.get_yaxis_transform(), ha="right", fontsize=4.9, color=INK2)
ax.set_yticks([0, 1]); ax.set_yticklabels(["Prostate (7)", "DLPFC (12)"]); ax.set_ylim(-0.5, 1.6)
ax.set_xlabel("Best spatial − GMM, mean ARI\n(one dot per left-out section)"); ax.set_title("g   Leave one section out", loc="left")
ax = fig.add_subplot(gs[2, 4:8]); gridy(ax)
for ds, mk in [("DLPFC", "o"), ("prostate", "s")]:
    d = main[main["dataset"] == ds]
    for seed in range(5):
        mm = d[d["seed"] == seed].groupby("method")["ari"].mean().sort_values(ascending=False)
        ax.scatter(seed + (-0.12 if ds == "DLPFC" else 0.12), list(mm.index).index("gmm") + 1, s=16, marker=mk, color=DSCOL[ds],
                   edgecolor=SURF, lw=0.4, zorder=3)
ax.set_xticks(range(5)); ax.set_xlabel("Seed"); ax.set_ylabel("Rank of GMM"); ax.set_ylim(len(METHODS) + 0.5, 0.5)
ax.set_yticks(list(range(1, len(METHODS) + 1, 2)))
ax.set_title("h   Rank of GMM per seed", loc="left")

# i: agreement between metrics (prostate)
ax = fig.add_subplot(gs[2, 8:12])
mets = ["ari", "nmi", "ami", "homogeneity", "completeness", "ari_coarse", "tumor_f1"]
Mm = main[main["dataset"] == "prostate"].groupby("method")[mets].mean()
C = Mm.corr(method="spearman").values
heat(ax, C, SEQ[:11], 0, 1, fmt="{:.2f}", fs=4.2, xt=["ARI", "NMI", "AMI", "homog.", "compl.", "coarse", "tumour"],
     yt=["ARI", "NMI", "AMI", "homog.", "compl.", "coarse", "tumour"])
ax.tick_params(axis="x", rotation=60, labelsize=5); ax.set_title("i   Prostate: metrics agree (ρ)", loc="left")

# j: bootstrap distributions of the difference from GMM
ax = fig.add_subplot(gs[3, 0:4]); gridx(ax); zero(ax, "x")
key = ["graphst", "smooth", "stagate", "sedr"]
for ds_i, ds in enumerate(["DLPFC", "prostate"]):
    P = main[main["dataset"] == ds].groupby(["method", "slide"])["ari"].mean().unstack()
    for i, m in enumerate(key):
        d = (P.loc[m] - P.loc["gmm"]).values
        bs = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(1)
        y = (len(key) - 1 - i) + (0.18 if ds == "DLPFC" else -0.18)
        parts = ax.violinplot(bs, positions=[y], vert=False, widths=0.32, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(DSCOL[ds]); pc.set_alpha(0.55); pc.set_edgecolor("none")
ax.set_yticks(range(len(key))); ax.set_yticklabels([NAME[m] for m in key][::-1])
ax.set_xlabel("Bootstrap mean ΔARI vs GMM"); ax.set_title("j   Bootstrap: top methods", loc="left")

# k: SpaGCN refinement
ax = fig.add_subplot(gs[3, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for ds in ["DLPFC", "prostate"]:
    a_ = S[(S["dataset"] == ds) & (S["method"] == "spagcn")].groupby("slide")["ari"].mean()
    b_ = S[(S["dataset"] == ds) & (S["method"] == "spagcn_refined")].groupby("slide")["ari"].mean()
    ax.scatter(a_, b_.loc[a_.index], s=14, color=DSCOL[ds], edgecolor=SURF, lw=0.4, zorder=3)
lim = [0, 0.75]; ax.plot(lim, lim, color=MUTED, lw=0.6, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("SpaGCN ARI"); ax.set_ylabel("SpaGCN ARI after refinement"); ax.set_title("k   SpaGCN refinement step", loc="left")

# l, m: the label-free indicator does not survive a fixed K
ax = fig.add_subplot(gs[3, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
for ds in ["DLPFC", "prostate"]:
    g = LF[LF["dataset"] == ds]
    ax.scatter(g["gmm_pas"], g["gain9"], s=14, color=DSCOL[ds], edgecolor=SURF, lw=0.4, zorder=3)
r0 = spearmanr(LF["gmm_pas"], LF["gain9"])
ax.text(0.03, 0.97, f"reference K: ρ = {r0[0]:.2f}, P = {r0[1]:.2g}", transform=ax.transAxes, va="top", fontsize=4.9, color=INK2)
ax.set_xlabel("PAS of GMM domains (reference K)"); ax.set_ylabel("ΔARI, spatial (mean) − GMM")
ax.set_title("l   Candidate label-free indicator", loc="left")
ax = fig.add_subplot(gs[4, 0:4]); gridy(ax); zero(ax)
rhos = [("ref. K", r0[0], r0[1])] + [(f"K = {k}", *spearmanr(FK[f"pas_k{k}"], FK["gain9"])) for k in (4, 6, 8)]
for i, (lbl, r, p) in enumerate(rhos):
    ax.bar(i, r, color=SEQ[7] if i == 0 else SEQ[4], width=0.6)
    ax.text(i, r + 0.03, f"P = {p:.2g}", ha="center", fontsize=4.8, color=INK2)
ax.set_xticks(range(len(rhos))); ax.set_xticklabels([x[0] for x in rhos]); ax.set_ylim(0, 0.9)
ax.set_ylabel("Spearman ρ with spatial gain"); ax.set_title("m   Gone at fixed K (dropped)", loc="left")

# n: runtime scaling; o: DLPFC control for the majority vote
ax = fig.add_subplot(gs[4, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ends = []
for m in ["gmm", "graphst", "sedr", "bayesspace"]:
    d = main[(main["method"] == m) & main["slide"].isin(SO.index)].groupby("slide")["sec"].median()
    n_ = SO.loc[d.index, "n_spots"]
    ax.scatter(n_, d, s=6, color=COL[m], alpha=0.7, lw=0)
    z = np.polyfit(np.log10(n_), np.log10(d), 1); xs = np.linspace(n_.min(), n_.max(), 10)
    ax.plot(xs, 10 ** np.polyval(z, np.log10(xs)), color=COL[m], lw=0.9)
    ax.text(xs[-1] * 1.04, 10 ** np.polyval(z, np.log10(xs[-1])), f"{NAME[m]} ({z[0]:.1f})", fontsize=4.6, va="center", color=INK2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(right=ax.get_xlim()[1] * 3.2)
ax.set_xticks([1000, 2000, 4000]); ax.set_xticklabels(["1,000", "2,000", "4,000"]); ax.minorticks_off()
ax.set_xlabel("Spots per section"); ax.set_ylabel("Seconds per run (median)"); ax.set_title("n   Runtime scaling (slope)", loc="left")
ax = fig.add_subplot(gs[4, 8:12]); gridy(ax)
ms = MS[MS["dataset"] == "DLPFC"]
for m in [x for x in METHODS if x != "gmm"] + ["gmm"]:
    g = ms[ms["method"] == m].sort_values("level")
    ax.plot(g["level"], g["mean_ari"], color=COL[m], lw=2.0 if m == "gmm" else 0.8, marker="o", ms=2, alpha=1 if m == "gmm" else 0.6)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["original", "1 ring", "2 rings"]); ax.set_xlabel("DLPFC labels (majority vote)")
ax.set_ylabel("Mean ARI, DLPFC"); ax.set_title("o   Control: DLPFC unchanged", loc="left")
hs = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=c, mec=SURF, label=l) for c, l in
      [(C_SPATIAL, LEG["spatial"]), (C_SIMPLE, LEG["simple"]), (C_NONSP, LEG["nonspatial"])]] + \
     [Line2D([], [], marker="o", ls="", ms=4.5, mfc=DSCOL[d], mec=SURF, label=DSNAME[d]) for d in ["DLPFC", "prostate"]]
fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=5, fontsize=5.6, handletextpad=0.2, columnspacing=1.0)
save(fig, "Fig10_robustness")
print("FIG7_DONE")
