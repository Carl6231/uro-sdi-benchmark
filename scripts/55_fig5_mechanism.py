"""55_fig5_mechanism.py - Figure 5: the granularity of the reference labels explains the ranking flip (panels a-o)."""
from fig_common import *
from collections import Counter
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

GS = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")); GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
CF = pd.read_csv(os.path.join(TAB, "class_f1.csv")); CR = pd.read_csv(os.path.join(TAB, "class_gain_vs_patchiness.csv"))
MS = pd.read_csv(os.path.join(TAB, "majority_filter.csv")); ML = pd.read_csv(os.path.join(TAB, "majority_filter_labels.csv"))
D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
rng = np.random.default_rng(0)
BDOT = Line2D([], [], marker="o", ls="", ms=1.6, mfc=INK, mec=INK, label="boundary spot (touches another label)")


def boundary(v, nb):
    s = pd.notna(v)
    return np.array([bool(s[i]) and any(s[j] and v[j] != v[i] for j in nb[i]) for i in range(len(v))])


def majority(v, nb):
    s = pd.notna(v); out = v.copy()
    for i in np.where(s)[0]:
        cnt = Counter([v[i]] + [v[j] for j in nb[i] if s[j]]); top = max(cnt.values())
        if cnt[v[i]] < top:
            out[i] = sorted(x for x, c in cnt.items() if c == top)[0]
    return out


def label_map(ax, xy, v, nb, labels=True):
    bnd = boundary(v, nb)
    spatial(ax, xy, label_colors(v), fig)
    ax.scatter(xy[bnd, 0], xy[bnd, 1], s=0.3, color=INK, lw=0, zorder=4)
    if labels:
        vc = pd.Series(v).value_counts(); direct_labels(ax, xy, v, [c for c in vc.index if vc[c] >= 20], fs=4.8)
    return bnd


def note(ax, txt):
    ax.text(0.5, -0.02, txt, transform=ax.transAxes, ha="center", va="top", fontsize=5.2, color=INK2)


fig = plt.figure(figsize=(7.4, 10.6))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.55, 0.95, 1.9, 1.5, 1.35], hspace=0.66, wspace=2.6)

# a-c: label maps with boundary spots
maps0 = []
for i, (k, ttl) in enumerate([("151673", "DLPFC 151673"), ("H1_2", "Prostate H1_2"), ("H1_5", "Prostate H1_5")]):
    sid = next(s for s in GS["slide"] if s.endswith(k))
    ax = fig.add_subplot(gs[0, i * 4:(i + 1) * 4]); maps0.append(ax)
    lab, xy = load_section(sid); nb, _ = bu.lattice_neighbors(xy.values)
    bnd = label_map(ax, xy.values, lab.values, nb)
    r = GS.set_index("slide").loc[sid]
    ax.set_title(f"{'abc'[i]}   {ttl}", loc="left")
    note(ax, f"{r['boundary_frac'] * 100:.0f}% boundary spots, {int(r['n_patches'])} patches")
legend_below(fig, maps0, [BDOT], gap=0.022, fs=5.4)

# d-f: label geometry per section (shared dataset rows)
for c0, col, xl, lt, tt, logx in [(0, "boundary_frac", "Share of boundary spots", "d", "Boundary spots", False),
                                  (4, "n_patches", "Label patches per section", "e", "Number of patches", True),
                                  (8, "spot_weighted_patch", "Patch size around a spot", "f", "Patch size", True)]:
    ax = fig.add_subplot(gs[1, c0:c0 + 4]); gridx(ax)
    for yy, ds in enumerate(["ccRCC", "prostate", "DLPFC"]):
        v = GS.loc[GS["dataset"] == ds, col].values
        ax.scatter(v, yy + rng.uniform(-0.16, 0.16, len(v)), s=8, color=DSCOL[ds], alpha=0.75, lw=0, zorder=3)
        ax.plot([np.median(v)] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=1.0, zorder=4)
        ax.text(np.median(v), yy + 0.34, f"{np.median(v):.2f}" if col == "boundary_frac" else f"{np.median(v):.0f}",
                ha="center", va="bottom", fontsize=4.8, color=INK2)
    if logx: ax.set_xscale("log")
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["ccRCC (TLS)", "Prostate", "DLPFC"] if c0 == 0 else [])
    ax.set_ylim(-0.5, 2.75); ax.set_xlabel(xl); ax.set_title(f"{lt}   {tt}", loc="left")

# g: boundary share per class; h: class-level gain; i: per-method correlation
ax = fig.add_subplot(gs[2, 0:4]); gridx(ax)
cls = GC[GC["dataset"].isin(["DLPFC", "prostate"])].groupby(["dataset", "cls"])["boundary_frac"].median().reset_index()
cls = cls.sort_values(["dataset", "boundary_frac"])
for i, r in enumerate(cls.itertuples()):
    ax.barh(i, r.boundary_frac, color=CLASS_COL.get(r.cls, MUTED), height=0.74, edgecolor=SURF, lw=0.3)
ax.set_yticks(range(len(cls))); ax.set_yticklabels([CLASS_NAME.get(c, c) for c in cls["cls"]], fontsize=4.9)
nd = int((cls["dataset"] == "DLPFC").sum()); ax.axhline(nd - 0.5, color=AXIS, lw=0.8)
ax.text(0.98, (nd - 1) / 2, "DLPFC", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.2, color=INK2)
ax.text(0.98, nd + (len(cls) - nd - 1) / 2, "Prostate", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.2, color=INK2)
ax.set_ylim(-0.6, len(cls) - 0.4); ax.set_xlabel("Boundary share (median of sections)"); ax.set_title("g   Patchiness per class", loc="left")
ax = fig.add_subplot(gs[2, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
W = CF.pivot_table(index=["slide", "dataset", "cls"], columns="method", values="f1").reset_index().merge(GC, on=["slide", "dataset", "cls"])
W["gain9"] = W[[m for m in SP9 if m in W]].mean(1) - W["gmm"]; W["log10_n"] = np.log10(W["n_spots"])
for ds in ["DLPFC", "prostate"]:
    g = W[W["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["gain9"], s=8, color=DSCOL[ds], alpha=0.75, edgecolor=SURF, lw=0.3, zorder=3)
try:
    import statsmodels.formula.api as smf
    fit = smf.mixedlm("gain9 ~ boundary_frac + log10_n", W, groups=W["slide"]).fit(reml=True)
    xs = np.linspace(W["boundary_frac"].min(), W["boundary_frac"].max(), 40)
    ax.plot(xs, fit.params["Intercept"] + fit.params["boundary_frac"] * xs + fit.params["log10_n"] * W["log10_n"].median(), color=INK, lw=1.1)
    pv = fit.pvalues["boundary_frac"]
    ax.text(0.97, 0.97, f"mixed model: slope {fit.params['boundary_frac']:.2f}\nP {'< 0.001' if pv < 0.001 else f'= {pv:.3f}'}, {len(W)} classes",
            transform=ax.transAxes, ha="right", va="top", fontsize=4.9, color=INK2)
except Exception as e:
    print("mixed model skipped", e)
ax.set_xlabel("Boundary share of the class"); ax.set_ylabel(f"ΔF1, spatial (mean of {NSP}) − GMM")
ax.set_title("h   Patchier classes gain less", loc="left")
ax = fig.add_subplot(gs[2, 8:12]); gridx(ax); zero(ax, "x")
cr = CR[CR["scope"] == "all"].set_index("method")
o = [m for m in cr.sort_values("rho_vs_boundary").index if m in SP9] + ["spatial9"]
for i, m in enumerate(o):
    r = cr.loc[m]; y = len(o) - 1 - i; c = COL.get(m, INK)
    ax.scatter(r["rho_vs_boundary"], y, s=16, color=c if r["p"] < 0.05 else SURF, edgecolor=c, lw=0.8, zorder=3)
ax.set_yticks(range(len(o))); ax.set_yticklabels([NAME.get(m, f"mean of {NSP}") for m in o][::-1], fontsize=5.2)
ax.yaxis.tick_right(); ax.set_ylim(-0.6, len(o) - 0.4)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, mfc=INK2, mec=INK2, label="P < 0.05"),
                   Line2D([], [], marker="o", ls="", ms=3.5, mfc=SURF, mec=INK2, label="P ≥ 0.05")], loc="center left", fontsize=5)
ax.set_xlabel("Spearman ρ (class gain vs patchiness)"); ax.set_title("i   Per method", loc="left")

# j-l: majority vote applied to the labels of one prostate section
ch = ML[(ML["dataset"] == "prostate") & (ML["level"] == 1)].set_index("slide")["changed"]
sid = ch.idxmax()
l0, xy = load_section(sid); xyv = xy.values
nb1, _ = bu.lattice_neighbors(xyv, 1.1); nb2, _ = bu.lattice_neighbors(xyv, 2.1)
maps3 = []
for i, (k, v) in enumerate([("original labels", l0.values), ("1-ring majority vote", majority(l0.values, nb1)),
                            ("2-ring majority vote", majority(l0.values, nb2))]):
    ax = fig.add_subplot(gs[3, i * 4:(i + 1) * 4]); maps3.append(ax)
    bnd = label_map(ax, xyv, v, nb1, labels=False)
    s = pd.notna(v)
    ax.set_title(f"{'jkl'[i]}   {short(sid)}, {k}", loc="left", fontsize=6.2)
    note(ax, f"{bnd[s].mean() * 100:.0f}% boundary" + (f", {np.mean(v[s] != l0.values[s]) * 100:.0f}% relabelled" if i else ""))
present = [c for c in l0.value_counts().index if l0.value_counts()[c] >= 5]
legend_below(fig, maps3, class_handles(present) + [BDOT], gap=0.022, fs=5.4)

# m: re-scoring; n: methods above GMM; o: per-section loss with full smoothing
ax = fig.add_subplot(gs[4, 0:4]); gridy(ax)
ms = MS[MS["dataset"] == "prostate"]
for m in [x for x in METHODS if x != "gmm"] + ["gmm"]:
    g = ms[ms["method"] == m].sort_values("level")
    lw = 2.0 if m == "gmm" else 1.3 if m in ("smooth", "graphst") else 0.7
    ax.plot(g["level"], g["mean_ari"], color=COL[m], lw=lw, marker="o", ms=2.2, alpha=1 if lw > 1 else 0.6, zorder=4 if m == "gmm" else 2)
for m in ["gmm", "graphst"]:
    g = ms[ms["method"] == m].sort_values("level"); ax.text(2.07, g["mean_ari"].iloc[-1], NAME[m], fontsize=4.9, va="center", color=INK2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["original", "1 ring", "2 rings"]); ax.set_xlim(-0.15, 2.55)
ax.set_xlabel("Prostate labels after majority vote"); ax.set_ylabel("Mean ARI"); ax.set_title("m   Re-scored on smoothed labels", loc="left")
ax = fig.add_subplot(gs[4, 4:8]); gridy(ax)
for j, ds in enumerate(["DLPFC", "prostate"]):
    m_ = MS[(MS["dataset"] == ds) & MS["method"].isin(SP9)]
    cnt = m_.groupby("level")["diff_vs_gmm"].apply(lambda x: int((x > 0).sum()))
    bfr = ML[ML["dataset"] == ds].groupby("level")["boundary_frac"].mean()
    for lev in [0, 1, 2]:
        x = lev + (j - 0.5) * 0.36
        ax.bar(x, cnt[lev], width=0.34, color=DSCOL[ds])
        ax.text(x, cnt[lev] + 0.2, f"{bfr[lev]:.2f}", ha="center", fontsize=4.5, color=INK2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["original", "1 ring", "2 rings"]); ax.set_ylim(0, 10)
ax.text(0.98, 0.97, "number on bar: boundary share", transform=ax.transAxes, ha="right", va="top", fontsize=4.6, color=INK2)
ax.set_ylabel(f"Methods above GMM (of {NSP})"); ax.set_xlabel("Labels after majority vote"); ax.set_ylim(0, NSP + 1)
ax.set_title("n   Smoother labels, spatial wins", loc="left")
ax = fig.add_subplot(gs[4, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
per = D[D["dataset"].isin(["DLPFC", "prostate"])].groupby(["dataset", "slide", "ring", "alpha"])["ari"].mean()
resp = (per.xs((1, 1.0), level=("ring", "alpha")) - per.xs((0, 0.0), level=("ring", "alpha"))).rename("d").reset_index()
resp = resp.merge(GS[["slide", "boundary_frac"]], on="slide")
for ds in ["DLPFC", "prostate"]:
    g = resp[resp["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["d"], s=13, color=DSCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
sel = resp[(resp["dataset"] == "prostate") & (resp["boundary_frac"] < 0.5)]
place_labels(ax, fig, sel["boundary_frac"].values, sel["d"].values, [short(s) for s in sel["slide"]], fs=4.8)
r_ = spearmanr(resp["boundary_frac"], resp["d"])
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("ΔARI at α = 1")
ax.set_title(f"o   Per section (ρ = {r_[0]:.2f}, P = {r_[1]:.2f})", loc="left")
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.5, mfc=DSCOL[d], mec=SURF, label=DSNAME[d]) for d in ["DLPFC", "prostate", "ccRCC"]]
           + category_handles()[:3], loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=6, fontsize=5.4, handletextpad=0.2, columnspacing=1.0)
save(fig, "Fig5_label_granularity")
print("FIG5_DONE")
