"""m6_fig6.py - main Figure 6: label-free check with marker programmes, scorecard and practical guide (panels a-n)."""
from mstyle import *
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import marker_programs as mp

TCOL = {"DLPFC": MUTED, "prostate": VIOLET, "ccRCC": INK, "kidney": GREEN, "breast": MAGENTA}
TNAME = {"DLPFC": "DLPFC", "prostate": "Prostate", "ccRCC": "ccRCC", "kidney": "Kidney", "breast": "Breast"}
PSHORT = {"Layer 1 (astrocytes)": "L1", "Neurons": "Neurons", "White matter": "WM", "Benign glands": "Benign",
          "Lymphocytes": "Lymph", "Prostate tumour": "Tumour", "Smooth-muscle stroma": "Stroma", "Collecting duct": "CD",
          "Endothelium": "Endo", "Glomeruli (podocytes)": "Glom", "Proximal tubule": "PT", "TLS (B cells, FDC)": "TLS",
          "Thick ascending limb": "TAL", "ccRCC tumour (hypoxia)": "Tumour", "Epithelium": "Epith", "Myoepithelium": "Myoepi",
          "Fibroblast stroma": "Stroma", "Adipocytes": "Adipo"}
rng = np.random.default_rng(3)

E = pd.read_csv(os.path.join(TAB, "marker_main_eta.csv"))
E = pd.concat([E, pd.read_csv(os.path.join(TAB, "marker_ext_eta.csv"))])
E = E[~((E["tissue"] == "ccRCC") & (E["k"] != 6)) & E["method"].isin(METHODS)]
EM = E.groupby(["slide", "tissue", "program", "method"])["eta2"].mean().unstack()
EM["sp"] = EM[[m for m in SP9 if m in EM]].mean(1)
D = pd.concat([pd.read_csv(os.path.join(bu.BASE, "results/dose", f)) for f in ("real_dose.csv", "ext_dose.csv")])
D["tissue"] = D["tissue"].fillna("none")
pc = [c for c in D.columns if c.startswith("eta2|")]
L = D.melt(id_vars=["slide", "tissue", "variant", "alpha", "seed"], value_vars=pc, var_name="program", value_name="eta2").dropna()
L["program"] = L["program"].str[5:]
LM = L.groupby(["slide", "tissue", "variant", "program", "alpha"])["eta2"].mean().unstack("alpha")
LD = LM.sub(LM[0.0], axis=0)
tissues = ["DLPFC", "prostate", "ccRCC", "kidney", "breast"]

fig = plt.figure(figsize=(FW, 9.6))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.3, 1.45, 1.25, 1.55, 1.25], hspace=0.66, wspace=2.2)

# a-d: kidney section - glomerular programme and the domains at increasing smoothing
kid = EM.xs("Glomeruli (podocytes)", level="program").reset_index()
kid = kid[kid["tissue"] == "kidney"]
nsp = pd.read_csv(os.path.join(TAB, "marker_grain_all.csv")).groupby("slide")["n_spots"].first()
KS = kid[kid["slide"].map(nsp) > 2000].sort_values("gmm", ascending=False)["slide"].iloc[0]
a_k = bu.load_slide(KS)
_, progs = mp.programs_for(KS)
s_glom, genes = mp.program_scores(a_k, {"Glomeruli (podocytes)": progs["Glomeruli (podocytes)"]})["Glomeruli (podocytes)"]
xy = np.asarray(a_k.obsm["spatial"], float)
z = np.clip((s_glom - s_glom.mean()) / s_glom.std(), -1, 3)
cmz = LinearSegmentedColormap.from_list("q", ["#f0efec"] + SEQ[3:])
maps0 = []
ax = fig.add_subplot(gs[0, 0:3]); maps0.append(ax)
spatial(ax, xy, [cmz((v + 1) / 4) for v in z], fig)
ptitle(ax, "a", "Kidney: podocyte genes", fontsize=FS)
Z = np.asarray(bu.standard_pca(a_k, seed=0).obsm["X_pca"], float)
nb, _ = bu.lattice_neighbors(xy, 1.1)
WZ = np.stack([Z[n].mean(0) if len(n) else Z[i] for i, n in enumerate(nb)])
preds = {al: bu.gmm_tied((1 - al) * Z + al * WZ, 6, 0) for al in (0.0, 0.5, 1.0)}
ref = preds[0.0]; dom_pal = ["#f3c06b", "#9ec5f4", "#a8dcc4", "#f2b5cd", "#d6d4cb", "#c4bdf3"]   # pastel: domains, not methods
topg = s_glom >= np.quantile(s_glom, 0.9)
for j, (al, lt) in enumerate([(0.0, "b"), (0.5, "c"), (1.0, "d")]):
    p = preds[al]
    C = np.array([[np.sum((p == u) & (ref == v)) for v in range(6)] for u in range(6)])
    r, c = linear_sum_assignment(-C); mpc = {u: dom_pal[v] for u, v in zip(r, c)}
    ax = fig.add_subplot(gs[0, 3 * (j + 1):3 * (j + 2)]); maps0.append(ax)
    spatial(ax, xy, [mpc[v] for v in p], fig)
    ax.scatter(xy[topg, 0], xy[topg, 1], s=1.6, color=INK, lw=0, zorder=4)
    ptitle(ax, lt, f"α = {al:g}  ·  η² {mp.eta2(s_glom, p):.2f}", fontsize=FS)
top_align(maps0)
maps_legend(fig, maps0, [Line2D([], [], marker="o", ls="", ms=5, mfc=SEQ[10], mec=SURF, label="high podocyte score"),
                         Line2D([], [], marker="o", ls="", ms=2.8, mfc=INK, mec=INK, label="top-decile podocyte spots"),
                         Line2D([], [], marker="o", ls="", ms=5, mfc="#9ec5f4", mec=SURF, label="domains (K = 6), GMM on smoothed PCs")],
            gap=0.006)

# e: delta eta2 vs GMM per method and programme; f: per section; g: per method
ax = fig.add_subplot(gs[1, 1:7])
cols = [(t, p) for t in tissues for p in sorted(EM.xs(t, level="tissue").index.get_level_values("program").unique())]
meths = [m for m in METHODS if m != "gmm" and m in EM]
Mx = np.array([[(EM.xs((t, p), level=("tissue", "program"))[m] - EM.xs((t, p), level=("tissue", "program"))["gmm"]).mean()
                for (t, p) in cols] for m in meths])
dcm = LinearSegmentedColormap.from_list("d", [DIV_NEG, DIV_MID, DIV_POS])
ax.imshow(Mx, cmap=dcm, vmin=-0.15, vmax=0.15, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(meths))); ax.set_yticklabels([NAME[m] for m in meths], fontsize=FMIN)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([PSHORT.get(p, p) for t, p in cols], rotation=90, fontsize=FMIN - 0.5)
for i, t in enumerate(tissues):
    js = [j for j, (tt, _) in enumerate(cols) if tt == t]
    ax.plot([min(js) - 0.4, max(js) + 0.4], [-0.95, -0.95], color=TCOL[t], lw=2.4, clip_on=False, solid_capstyle="butt")
    ax.text(np.mean(js), -1.5, TNAME[t], ha="center", va="bottom", fontsize=FMIN, color=INK2, clip_on=False)
    if i:
        ax.axvline(min(js) - 0.5, color=SURF, lw=1.4)
for s_ in ax.spines.values(): s_.set_visible(False)
ptitle(ax, "e", "Δη² vs GMM (blue: spatial better)", pad=15)
ax = fig.add_subplot(gs[1, 7:9]); gridx(ax); zero(ax, "x")
sec = (EM["sp"] - EM["gmm"]).groupby(level=["slide", "tissue"]).mean().reset_index(name="d")
for i, t in enumerate(tissues):
    v = sec.loc[sec["tissue"] == t, "d"].values; y = len(tissues) - 1 - i
    ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=8, color=TCOL[t], alpha=0.7, lw=0)
    ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.1)
ax.set_yticks(range(len(tissues))); ax.set_yticklabels([TNAME[t] for t in tissues][::-1], fontsize=FMIN)
ax.set_xlabel("Δη², sections"); ptitle(ax, "f", "Per section")
ax = fig.add_subplot(gs[1, 9:12]); gridx(ax); zero(ax, "x")
order = sorted(meths, key=lambda m: -(EM[m] - EM["gmm"]).groupby(level="tissue").mean().drop("DLPFC").mean())
for i, m in enumerate(order):
    y = len(order) - 1 - i
    dd = (EM[m] - EM["gmm"]).groupby(level="tissue").mean()
    oth = dd.drop("DLPFC").mean(); dl = dd["DLPFC"]
    ax.plot([dl, oth], [y, y], color=GRID, lw=1.4, zorder=1)
    ax.scatter(dl, y, s=12, color=MUTED, zorder=3); ax.scatter(oth, y, s=14, color=COL[m], zorder=3)
method_rows_m(ax, order, fs=FMIN)
ax.yaxis.tick_right()
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(3))
ax.set_xlabel("Δη² (grey: DLPFC)"); ptitle(ax, "g", "Per method")

# h: kidney dose with held-out genes; i: not circular; j: agrees with labels
ax = fig.add_subplot(gs[2, 0:4]); gridy(ax); zero(ax)
for t in ["DLPFC", "prostate", "breast", "kidney"]:
    sub = LD.xs(("heldout",), level=("variant",)).xs(t, level="tissue")
    cur = sub.groupby(level="alpha", axis=1).mean().mean() if False else sub.mean()
    ax.plot(cur.index, cur.values, color=TCOL[t], lw=1.6, marker="o", ms=2.4, mec=SURF, mew=0.4)
    ax.text(1.03, cur.values[-1], TNAME[t], fontsize=FMIN, color=INK2, va="center")
ax.set_xlim(-0.03, 1.3); ax.set_xticks([0, 0.5, 1.0]); ax.set_xlabel("Neighbour weight α"); ax.set_ylabel("Δη² (held-out genes)")
ptitle(ax, "h", "Smoothing dose, no labels")
ax = fig.add_subplot(gs[2, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
w = LD[1.0].unstack("variant").dropna()
for t in tissues:
    if t in w.index.get_level_values("tissue"):
        g = w.xs(t, level="tissue"); ax.scatter(g["all"], g["heldout"], s=6, color=TCOL[t], alpha=0.6, lw=0)
lim = [w.min().min() - 0.02, w.max().max() + 0.02]
ax.plot(lim, lim, color=MUTED, lw=0.6, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
r_ = spearmanr(w["all"], w["heldout"])
ax.set_xlabel("Δη², all genes"); ax.set_ylabel("Δη², genes held out")
ptitle(ax, "i", f"Not circular (ρ = {r_[0]:.2f})")
ax = fig.add_subplot(gs[2, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax); zero(ax, "x")
de = LD[1.0].xs("heldout", level="variant").groupby(level=["slide", "tissue"]).mean().rename("deta").reset_index()
ar = D[(D["variant"] == "all") & D["ari"].notna()].groupby(["slide", "alpha"])["ari"].mean().unstack()
de["dari"] = de["slide"].map(ar[1.0] - ar[0.0])
k_ = de[de["tissue"].isin(["DLPFC", "prostate", "breast"]) & de["dari"].notna()]
for t, g in k_.groupby("tissue"):
    ax.scatter(g["deta"], g["dari"], s=14, color=TCOL[t], edgecolor=SURF, lw=0.4, zorder=3)
r_ = spearmanr(k_["deta"], k_["dari"])
ax.set_xlabel("Δη² at α = 1 (no labels)"); ax.set_ylabel("ΔARI at α = 1")
ptitle(ax, "j", f"Agrees with labels (ρ = {r_[0]:.2f})")

# k: scorecard; l: mean rank
SC = pd.read_csv(os.path.join(TAB, "guide_scorecard.csv"), index_col=0)
crit = [c for c in SC.columns if c not in ("GPU", "mean_rank_accuracy")]
LOWER = {"Seed SD", "Runtime (median)"}
Zs = SC[crit].astype(float).copy()
for c in crit:
    v = np.log10(Zs[c]) if c.startswith("Runtime") else Zs[c]
    z_ = (v - v.min()) / (v.max() - v.min())
    Zs[c] = 1 - z_ if c in LOWER else z_
HDR = {"DLPFC ARI": "DLPFC\nARI", "Prostate ARI": "Prostate\nARI", "Prostate tumour F1": "Tumour\nF1", "ccRCC TLS F1": "TLS\nF1",
       "Breast ARI": "Breast\nARI", "Simulated fine grain": "Sim.\nfine", "Simulated coarse grain": "Sim.\ncoarse",
       "Marker η² vs GMM": "Marker\nΔη²", "Seed SD": "Seed\nSD", "Runtime (median)": "Run\ntime"}
ax = fig.add_subplot(gs[3, 0:8])
A = np.column_stack([Zs.values, np.full(len(Zs), np.nan)])
heat(ax, A, SEQ[:11], 0, 1, text=False, xt=[HDR[c] for c in crit] + ["GPU"], yt=[NAME[m] for m in SC.index])
for i, m in enumerate(SC.index):
    for j, c in enumerate(crit + ["GPU"]):
        if c == "GPU":
            txt, zz = ("yes" if str(SC.loc[m, "GPU"]) == "True" else "–"), 0
        else:
            v = SC.loc[m, c]; zz = Zs.loc[m, c]
            txt = (f"{v:.0f} s" if v < 90 else f"{v / 60:.0f} min") if c.startswith("Runtime") else (f"{v:+.2f}" if c.startswith("Marker")
                                                                                                  else f"{v:.2f}")
        ax.text(j, i, txt, ha="center", va="center", fontsize=FMIN - 0.2, color=SURF if zz > 0.62 else INK)
ax.xaxis.tick_top(); ax.tick_params(axis="x", labelsize=FMIN - 0.3, pad=1); ax.tick_params(axis="y", labelsize=FMIN)
for lbl in ax.get_yticklabels():
    lbl.set_color(COL.get({v: k for k, v in NAME.items()}.get(lbl.get_text(), ""), INK))
ptitle(ax, "k", "Scorecard (darker = better within column)", pad=20)
ax = fig.add_subplot(gs[3, 9:12]); gridx(ax)
rank = SC["mean_rank_accuracy"]
for i, m in enumerate(SC.index):
    ax.barh(len(SC) - 1 - i, rank[m], color=COL[m], height=0.62)
    ax.text(rank[m] + 0.2, len(SC) - 1 - i, f"{rank[m]:.1f}", va="center", fontsize=FMIN, color=INK2)
ax.set_yticks([]); ax.set_ylim(-0.6, len(SC) - 0.4); ax.set_xlim(0, 12.5)
ax.set_xlabel("Mean rank, 8 accuracy\ncolumns (1 = best)")
ptitle(ax, "l", "Overall", pad=20)

# m: break-even line; n: decision guide
BE = pd.read_csv(os.path.join(TAB, "guide_breakeven.csv"))
ax = fig.add_subplot(gs[4, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.fill_between(BE["n_spots"], BE["ci_lo"].clip(0, 1), BE["ci_hi"].clip(0, 1), color=SEQ[1], lw=0)
ax.plot(BE["n_spots"], BE["break_even"].clip(0, 1), color=SEQ[10], lw=1.6)
RT = pd.read_csv(os.path.join(TAB, "ext_rule_test.csv"))
CF = pd.read_csv(os.path.join(TAB, "class_f1.csv")); GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
Wc = CF[CF["dataset"].isin(["DLPFC", "prostate"])].pivot_table(index=["slide", "dataset", "cls"], columns="method", values="f1")
Wc = Wc.reset_index().merge(GC, on=["slide", "dataset", "cls"]); Wc["gain"] = Wc[[m for m in SP9 if m in Wc]].mean(1) - Wc["gmm"]
for g, ds, col in [(Wc[Wc["dataset"] == "DLPFC"], "DLPFC", "gain"), (Wc[Wc["dataset"] == "prostate"], "prostate", "gain"), (RT, "breast", "gain9")]:
    up = g[col] > 0
    ax.scatter(g.loc[up, "n_spots"], g.loc[up, "boundary_frac"], s=9, color=TCOL[ds], lw=0, zorder=3)
    ax.scatter(g.loc[~up, "n_spots"], g.loc[~up, "boundary_frac"], s=9, color=SURF, edgecolor=TCOL[ds], lw=0.6, zorder=3)
ax.set_xscale("log"); ax.set_xlim(30, 6000); ax.set_ylim(0, 1)
ax.text(0.04, 0.05, "spatial helps", transform=ax.transAxes, fontsize=FS, color=SEQ[10], fontweight="bold",
        bbox=dict(fc=SURF, ec="none", pad=0.8, alpha=0.85), zorder=6)
ax.text(0.96, 0.9, "GMM as good", transform=ax.transAxes, fontsize=FS, color=INK2, ha="right", fontweight="bold",
        bbox=dict(fc=SURF, ec="none", pad=0.8, alpha=0.85), zorder=6)
ax.set_xlabel("Class size (spots)"); ax.set_ylabel("Boundary share of the class")
ptitle(ax, "m", "Break-even line (● gain > 0)")
b100, b500, b2000 = [float(BE.loc[BE["n_spots"] == n, "break_even"].iloc[0]) for n in (100, 500, 2000)]
top3 = [NAME[m] for m in SC.index if m not in ("gmm", "leiden")][:3]
ax = fig.add_subplot(gs[4, 4:12]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
boxes = [(0.0, 0.08, 0.215, 0.8, "1  Target", "domain size and\nboundary share,\nfrom histology\nor prior work", "#f3f2ee"),
         (0.265, 0.08, 0.215, 0.8, "2  Break-even", f"read it off m:\n{b100:.2f} at 100 spots\n{b500:.2f} at 500\n{b2000:.2f} at 2,000", "#f3f2ee"),
         (0.53, 0.52, 0.215, 0.43, "Below the line", f"spatial: {top3[0]}\nor {top3[1]}", SEQ[1]),
         (0.53, 0.0, 0.215, 0.43, "Above the line", "GMM, or light\nsmoothing (α ≈ 0.1)", "#fde3cf"),
         (0.795, 0.08, 0.205, 0.8, "3  Check", "marker η² with\nheld-out genes vs\nGMM; a drop flags\nlost structure", "#f3f2ee")]
for x, y, wd, ht, hd, tx, fc in boxes:
    ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec=AXIS, lw=0.7))
    ax.text(x + 0.012, y + ht - 0.07, hd, fontsize=FS, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.012, y + ht - (0.25 if ht > 0.5 else 0.2), tx, fontsize=FMIN, color=INK2, va="top", linespacing=1.3)
arr = dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7)
ax.annotate("", xy=(0.265, 0.48), xytext=(0.215, 0.48), arrowprops=arr)
ax.annotate("", xy=(0.53, 0.74), xytext=(0.48, 0.6), arrowprops=arr)
ax.annotate("", xy=(0.53, 0.22), xytext=(0.48, 0.36), arrowprops=arr)
ax.annotate("", xy=(0.795, 0.6), xytext=(0.745, 0.74), arrowprops=arr)
ax.annotate("", xy=(0.795, 0.36), xytext=(0.745, 0.22), arrowprops=arr)
ptitle(ax, "n", "Decision guide")
top_legend(fig, [Line2D([], [], marker="o", ls="", ms=5, mfc=TCOL[t], mec=SURF, label=TNAME[t] + (" (unlabelled)" if t == "kidney" else ""))
                 for t in tissues] + cat_handles(), y=0.935, ncol=4, cs=1.0)
msave(fig, "Figure6")
print("M6_DONE", KS)
