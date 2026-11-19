"""80_fig9_markers.py - Figure 9: label-free check with marker-gene programmes (panels a-o).
Question: do spatial domains keep or lose biologically defined expression programmes, without using any annotation?
eta^2 = share of a programme's spot-level variance explained by the domains. Dose analysis: PCA-smooth alpha sweep with
the programme genes removed from the clustering features (held-out) or kept (all genes).
"""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
import marker_programs as mp

TCOL = {"DLPFC": MUTED, "prostate": VIOLET, "ccRCC": INK, "kidney": GREEN, "breast": MAGENTA}
TNAME = {"DLPFC": "DLPFC", "prostate": "Prostate cancer", "ccRCC": "ccRCC", "kidney": "Kidney (unlabelled)", "breast": "Breast cancer"}
SPM = [m for m in SP9]
rng = np.random.default_rng(3)
PSHORT = {"Layer 1 (astrocytes)": "L1", "Neurons": "Neurons", "White matter": "WM", "Benign glands": "Benign",
          "Lymphocytes": "Lymph", "Prostate tumour": "Tumour", "Smooth-muscle stroma": "Stroma", "Collecting duct": "CD",
          "Endothelium": "Endo", "Glomeruli (podocytes)": "Glom", "Proximal tubule": "PT", "TLS (B cells, FDC)": "TLS",
          "Thick ascending limb": "TAL", "ccRCC tumour (hypoxia)": "Tumour", "Epithelium": "Epith", "Myoepithelium": "Myoepi",
          "Fibroblast stroma": "Stroma", "Adipocytes": "Adipo"}

# ---------------- data ----------------
E = pd.read_csv(os.path.join(TAB, "marker_main_eta.csv"))
if os.path.exists(os.path.join(TAB, "marker_ext_eta.csv")):
    E = pd.concat([E, pd.read_csv(os.path.join(TAB, "marker_ext_eta.csv"))])
E = E[~((E["tissue"] == "ccRCC") & (E["k"] != 6)) & E["method"].isin(METHODS)]
EM = E.groupby(["slide", "tissue", "program", "method"])["eta2"].mean().unstack()
EM["sp"] = EM[[m for m in SPM if m in EM]].mean(1)
D = pd.concat([pd.read_csv(os.path.join(bu.BASE, "results/dose", f)) for f in ("real_dose.csv", "ext_dose.csv")
               if os.path.exists(os.path.join(bu.BASE, "results/dose", f))])
D["tissue"] = D["tissue"].fillna("none")
pc = [c for c in D.columns if c.startswith("eta2|")]
L = D.melt(id_vars=["slide", "tissue", "variant", "alpha", "seed"], value_vars=pc, var_name="program", value_name="eta2").dropna()
L["program"] = L["program"].str[5:]
LM = L.groupby(["slide", "tissue", "variant", "program", "alpha"])["eta2"].mean().unstack("alpha")
LD = LM.sub(LM[0.0], axis=0)                                   # change vs alpha = 0
CO = pd.read_csv(os.path.join(TAB, "expression_coherence.csv")).set_index("slide")
tissues = [t for t in ["DLPFC", "prostate", "ccRCC", "kidney", "breast"] if t in EM.index.get_level_values("tissue")]

fig = plt.figure(figsize=(7.4, 10.4))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.35, 1.45, 1.25, 1.3, 1.35], hspace=0.75, wspace=2.4)


def score_map(ax, sid, prog, title_txt):
    a = bu.load_slide(sid)
    _, progs = mp.programs_for(sid)
    s, genes = mp.program_scores(a, {prog: progs[prog]})[prog]
    z = np.clip((s - s.mean()) / s.std(), -1, 3)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("q", ["#f0efec"] + SEQ[3:])
    spatial(ax, np.asarray(a.obsm["spatial"], float), [cm((v + 1) / 4) for v in z], fig)
    ax.set_title(title_txt, loc="left", fontsize=6.2)
    ax.text(0.0, -0.04, ", ".join(genes[:5]) + ("..." if len(genes) > 5 else ""), transform=ax.transAxes, fontsize=4.6,
            color=INK2, va="top")
    return a, s


# a-c: programme maps
kid = EM.xs("Glomeruli (podocytes)", level="program").reset_index()
kid = kid[kid["tissue"] == "kidney"]
nsp = pd.read_csv(os.path.join(TAB, "marker_grain_all.csv")).groupby("slide")["n_spots"].first()
kid = kid[kid["slide"].map(nsp) > 2000].sort_values("gmm", ascending=False)
KS = kid["slide"].iloc[0]
ax = fig.add_subplot(gs[0, 0:4]); a_k, s_glom = score_map(ax, KS, "Glomeruli (podocytes)", f"a   Kidney {short(KS)}: glomeruli")
ax = fig.add_subplot(gs[0, 4:8]); score_map(ax, KS, "Proximal tubule", "b   Same section: proximal tubule")
PS = next(s for s in bu.list_slides("Erickson") if s.endswith("H1_2"))
ax = fig.add_subplot(gs[0, 8:12]); score_map(ax, PS, "Prostate tumour", "c   Prostate H1_2: tumour programme")
from matplotlib.colors import LinearSegmentedColormap as _LSC
cax = ax.inset_axes([0.72, -0.2, 0.28, 0.04])
cbm = fig.colorbar(plt.cm.ScalarMappable(cmap=_LSC.from_list("q", ["#f0efec"] + SEQ[3:]), norm=plt.Normalize(-1, 3)), cax=cax,
                   orientation="horizontal")
cbm.outline.set_visible(False); cbm.ax.tick_params(labelsize=4.5); cbm.set_ticks([-1, 1, 3])
cax.set_title("programme score (z)", fontsize=4.6, color=INK2, pad=2, loc="center")

# d: delta eta2 vs GMM per method and programme (main runs)
ax = fig.add_subplot(gs[1, 0:6])
cols = [(t, p) for t in tissues for p in sorted(EM.xs(t, level="tissue").index.get_level_values("program").unique())]
meths = [m for m in METHODS if m != "gmm" and m in EM]
M = np.array([[(EM.xs((t, p), level=("tissue", "program"))[m] - EM.xs((t, p), level=("tissue", "program"))["gmm"]).mean()
               for (t, p) in cols] for m in meths])
from matplotlib.colors import LinearSegmentedColormap
dcm = LinearSegmentedColormap.from_list("d", [DIV_NEG, DIV_MID, DIV_POS])
ax.imshow(M, cmap=dcm, vmin=-0.15, vmax=0.15, aspect="auto", interpolation="nearest")
ax.set_yticks(range(len(meths))); ax.set_yticklabels([NAME[m] for m in meths], fontsize=5.0)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([PSHORT.get(p, p) for t, p in cols], rotation=90, fontsize=4.8)
for i, t in enumerate(tissues):
    js = [j for j, (tt, _) in enumerate(cols) if tt == t]
    ax.plot([min(js) - 0.4, max(js) + 0.4], [-0.9, -0.9], color=TCOL[t], lw=2.0, clip_on=False, solid_capstyle="butt")
    ax.text(np.mean(js), -1.4, TNAME[t].split(" (")[0], ha="center", va="bottom", fontsize=4.8, color=INK2, clip_on=False)
    if i:
        ax.axvline(min(js) - 0.5, color=SURF, lw=1.2)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title("d   Δη² vs GMM, benchmark runs", loc="left", pad=14)
cax = ax.inset_axes([0.62, -0.42, 0.36, 0.045])
sm = plt.cm.ScalarMappable(cmap=dcm, norm=plt.Normalize(-0.15, 0.15))
cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=4.5)
cb.set_ticks([-0.15, 0, 0.15]); cax.set_title("Δη² (red: GMM better)", fontsize=4.6, color=INK2, pad=2, loc="center")

# e: per tissue, section-level delta (mean of 9 spatial - GMM, averaged over programmes)
ax = fig.add_subplot(gs[1, 6:9]); gridx(ax); zero(ax, "x")
sec = (EM["sp"] - EM["gmm"]).groupby(level=["slide", "tissue"]).mean().reset_index(name="d")
for i, t in enumerate(tissues):
    v = sec.loc[sec["tissue"] == t, "d"].values; y = len(tissues) - 1 - i
    ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=7, color=TCOL[t], alpha=0.6, lw=0)
    ax.plot([np.median(v)] * 2, [y - 0.3, y + 0.3], color=INK, lw=1.0)
ax.set_yticks(range(len(tissues))); ax.set_yticklabels([TNAME[t] for t in tissues][::-1], fontsize=5.0)
ax.set_xlabel("Δη², spatial (mean of 9) − GMM"); ax.set_title("e   Per section", loc="left")

# f: per method, DLPFC vs all other tissues
ax = fig.add_subplot(gs[1, 9:12]); gridx(ax); zero(ax, "x")
order = sorted(meths, key=lambda m: -(EM[m] - EM["gmm"]).groupby(level="tissue").mean().drop("DLPFC", errors="ignore").mean())
for i, m in enumerate(order):
    y = len(order) - 1 - i
    dd = (EM[m] - EM["gmm"]).groupby(level="tissue").mean()
    oth = dd.drop("DLPFC", errors="ignore").mean(); dl = dd.get("DLPFC", np.nan)
    ax.plot([dl, oth], [y, y], color=GRID, lw=1.2, zorder=1)
    ax.scatter(dl, y, s=11, color=MUTED, zorder=3); ax.scatter(oth, y, s=11, color=COL[m], zorder=3)
method_rows(ax, order, fs=5.0)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(3))
ax.set_xlabel("Δη² vs GMM (gray: DLPFC)"); ax.set_title("f   Per method", loc="left")

# g-i: dose curves (held-out genes), mean change vs alpha per programme
pal = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
for c0, t, lt in [(0, "DLPFC", "g"), (4, "prostate", "h"), (8, "kidney", "i")]:
    ax = fig.add_subplot(gs[2, c0:c0 + 4]); gridy(ax); zero(ax)
    sub = LD.xs(("heldout",), level=("variant",)).xs(t, level="tissue") if t in LD.index.get_level_values("tissue") else None
    if sub is None:
        continue
    progs = sorted(sub.index.get_level_values("program").unique())
    ends = []
    for j, p in enumerate(progs):
        cur = sub.xs(p, level="program").mean()
        ax.plot(cur.index, cur.values, color=pal[j % 8], lw=1.1, marker="o", ms=1.8)
        ends.append((cur.values[-1], p, pal[j % 8]))
    allv = LD.xs(("all",), level=("variant",)).xs(t, level="tissue").mean()
    ax.plot(allv.index, allv.values, color=INK, lw=0.9, ls="--")
    ends.sort()
    ys = [e[0] for e in ends]
    lo, hi = ax.get_ylim(); gap = (hi - lo) * 0.075
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    for (v, p, c), yy in zip(ends, ys):
        ax.text(1.03, yy, PSHORT.get(p, p), fontsize=4.6, color=INK2, va="center")
    ax.set_xlim(-0.03, 1.32); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Neighbour weight α"); ax.set_ylabel("Δη² vs α = 0 (held-out genes)" if c0 == 0 else "")
    ax.set_title(f"{lt}   {TNAME[t].split(' (')[0]}: smoothing dose", loc="left")

# j: held-out vs all genes
ax = fig.add_subplot(gs[3, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
w = LD[1.0].unstack("variant").dropna()
for t in tissues:
    g = w.xs(t, level="tissue") if t in w.index.get_level_values("tissue") else None
    if g is not None:
        ax.scatter(g["all"], g["heldout"], s=5, color=TCOL[t], alpha=0.6, lw=0)
lim = [w.min().min() - 0.02, w.max().max() + 0.02]
ax.plot(lim, lim, color=MUTED, lw=0.6, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
r_ = spearmanr(w["all"], w["heldout"])
ax.set_xlabel("Δη² at α = 1, all genes"); ax.set_ylabel("Δη² at α = 1, programme genes held out")
ax.set_title(f"j   Not circular (ρ = {r_[0]:.2f})", loc="left")

# k: label-free vs label-based (sections with labels)
ax = fig.add_subplot(gs[3, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax); zero(ax, "x")
de = LD[1.0].xs("heldout", level="variant").groupby(level=["slide", "tissue"]).mean().rename("deta").reset_index()
ar = D[(D["variant"] == "all") & D["ari"].notna()].groupby(["slide", "alpha"])["ari"].mean().unstack()
de["dari"] = de["slide"].map(ar[1.0] - ar[0.0])
k_ = de[de["tissue"].isin(["DLPFC", "prostate", "breast"]) & de["dari"].notna()]
for t, g in k_.groupby("tissue"):
    ax.scatter(g["deta"], g["dari"], s=12, color=TCOL[t], edgecolor=SURF, lw=0.4, zorder=3)
r_ = spearmanr(k_["deta"], k_["dari"])
ax.set_xlabel("Δη² at α = 1 (label-free)"); ax.set_ylabel("ΔARI at α = 1 (labels)")
ax.set_title(f"k   Agrees with labels (ρ = {r_[0]:.2f})", loc="left")

# l: coherence of expression
ax = fig.add_subplot(gs[3, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
de["coh"] = de["slide"].map(CO["moran_pc_w"])
for t, g in de.groupby("tissue"):
    ax.scatter(g["coh"], g["deta"], s=9, color=TCOL.get(t, INK), edgecolor=SURF, lw=0.3, zorder=3)
r_ = spearmanr(de["coh"], de["deta"], nan_policy="omit")
ax.set_xlabel("Expression coherence (Moran's I of PCs)"); ax.set_ylabel("Δη² at α = 1")
ax.set_title(f"l   vs expression coherence (ρ = {r_[0]:.2f})", loc="left")

# m-o: kidney domains with increasing smoothing; glomerular top-decile spots as black dots
Z = np.asarray(bu.standard_pca(a_k, seed=0).obsm["X_pca"], float)
xy = np.asarray(a_k.obsm["spatial"], float)
nb, _ = bu.lattice_neighbors(xy, 1.1)
WZ = np.stack([Z[n].mean(0) if len(n) else Z[i] for i, n in enumerate(nb)])
preds = {al: bu.gmm_tied((1 - al) * Z + al * WZ, 6, 0) for al in (0.0, 0.5, 1.0)}
ref = preds[0.0]; dom_pal = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
topg = s_glom >= np.quantile(s_glom, 0.9)
map_axes = []
for j, (al, lt) in enumerate([(0.0, "m"), (0.5, "n"), (1.0, "o")]):
    p = preds[al]
    C = np.array([[np.sum((p == u) & (ref == v)) for v in range(6)] for u in range(6)])
    r, c = linear_sum_assignment(-C); mpc = {u: dom_pal[v] for u, v in zip(r, c)}
    ax = fig.add_subplot(gs[4, j * 4:(j + 1) * 4]); map_axes.append(ax)
    spatial(ax, xy, [mpc[v] for v in p], fig)
    ax.scatter(xy[topg, 0], xy[topg, 1], s=1.2, color=INK, lw=0, zorder=4)
    ax.set_title(f"{lt}   α = {al:g}: η²(glomeruli) = {mp.eta2(s_glom, p):.2f}", loc="left", fontsize=6.2)
legend_below(fig, map_axes, [Line2D([], [], marker="o", ls="", ms=2.5, mfc=INK, mec=INK, label="top-decile glomerular programme"),
                              Line2D([], [], marker="o", ls="", ms=4.5, mfc=BLUE, mec=SURF, label="domains (K = 6, PCA-smooth GMM)")],
             gap=0.006)
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=4.5, mfc=TCOL[t], mec=SURF, label=TNAME[t]) for t in tissues]
           + category_handles()[:3], loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=4, fontsize=5.4, handletextpad=0.2, columnspacing=1.0)
# tables for the text
sec.to_csv(os.path.join(TAB, "marker_section_delta.csv"), index=False)
de.to_csv(os.path.join(TAB, "marker_dose_section.csv"), index=False)
save(fig, "Fig9_label_free_markers")
print("FIG9_DONE", KS)
