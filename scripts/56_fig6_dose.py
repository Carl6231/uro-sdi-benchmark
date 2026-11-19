"""56_fig6_dose.py - Figure 6: dose-response of spatial smoothing (panels a-o)."""
from fig_common import *
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, t as tdist
from sklearn.metrics import adjusted_rand_score
import scipy.sparse as sp

S = scores(); SD = scores("results/scores_dose.csv")
D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
GS = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")).set_index("slide")
LP = pd.read_csv(os.path.join(TAB, "gain_vs_label_fragmentation.csv"))
DIV = LinearSegmentedColormap.from_list("div", [DIV_NEG, DIV_MID, DIV_POS])
NSEC = {"DLPFC": 12, "prostate": 7}


def ring_table(ds, ring, met="ari"):
    d = D[(D["dataset"] == ds) & D[met].notna()]
    sub = pd.concat([d[d["alpha"] == 0].assign(ring=ring), d[d["ring"] == ring]])
    return sub.groupby(["slide", "alpha"])[met].mean().unstack()


def curve(ax, w, base, color, ls="-", band=True, lw=1.4, x=None):
    dl = w.sub(w[base], axis=0); m = dl.mean(); n = dl.notna().sum()
    h = tdist.ppf(0.975, n - 1) * dl.std(ddof=1) / np.sqrt(n)
    xx = np.asarray(x if x is not None else dl.columns, float)
    if band:
        ax.fill_between(xx, (m - h).values, (m + h).values, color=color, alpha=0.13, lw=0)
    ax.plot(xx, m.values, color=color, ls=ls, lw=lw, marker="o" if band else None, ms=2.4, mec=SURF, mew=0.4, zorder=3)


def divheat(ax, M, vmax, rows, cols, fs=3.9):
    A = np.asarray(M, float)
    ax.imshow(A, cmap=DIV, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=fs,
                    color=SURF if abs(A[i, j]) > 0.55 * vmax else INK)
    ax.set_xticks(range(A.shape[1])); ax.set_xticklabels(cols); ax.set_yticks(range(A.shape[0])); ax.set_yticklabels(rows)
    for s_ in ax.spines.values(): s_.set_visible(False)


fig = plt.figure(figsize=(7.4, 10.6))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.3, 1.5, 1.75, 1.3, 1.8], hspace=0.8, wspace=2.3)

# a-c: PCA-smooth by neighbourhood size
for i, ring in enumerate([1, 2, 3]):
    ax = fig.add_subplot(gs[0, i * 4:(i + 1) * 4]); gridy(ax); zero(ax)
    for ds in ["DLPFC", "prostate"]:
        curve(ax, ring_table(ds, ring), 0.0, DSCOL[ds])
    ax.set_ylim(-0.3, 0.2); ax.set_xlabel("Neighbour weight α")
    if i == 0: ax.set_ylabel("PCA-smooth: ΔARI vs α = 0")
    else: ax.set_yticklabels([])
    ax.set_title(f"{'abc'[i]}   {ring} ring{'s' if ring > 1 else ''} ({[6, 18, 36][i]} neighbours)", loc="left")

# d, e: per-section response (1 ring)
for c0, c1, ds, lt in [(0, 5, "DLPFC", "d"), (5, 9, "prostate", "e")]:
    ax = fig.add_subplot(gs[1, c0:c1])
    w = ring_table(ds, 1); dl = w.sub(w[0.0], axis=0).drop(columns=0.0)
    dl = dl.loc[GS.loc[dl.index, "boundary_frac"].sort_values().index]
    divheat(ax, dl.values, 0.3, [short(s) for s in dl.index], [f"{a:.1f}".replace("0.", ".") for a in dl.columns])
    ax.set_xlabel("α (1 ring)"); ax.set_title(f"{lt}   {'Prostate' if ds == 'prostate' else ds}: ΔARI per section", loc="left")

# f: best alpha vs boundary share
ax = fig.add_subplot(gs[1, 9:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
pts = []
for ds in ["DLPFC", "prostate"]:
    w = ring_table(ds, 1)
    best = w.idxmax(1)
    bf = GS.loc[best.index, "boundary_frac"]
    jit = np.random.default_rng(1).uniform(-0.018, 0.018, len(best))
    ax.scatter(bf, best.values + jit, s=14, color=DSCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
    pts.append(pd.DataFrame({"bf": bf.values, "best": best.values}))
pp = pd.concat(pts); r_ = spearmanr(pp["bf"], pp["best"])
ax.text(0.97, 0.97, f"ρ = {r_[0]:.2f}, P = {r_[1]:.2g}", transform=ax.transAxes, ha="right", va="top", fontsize=5, color=INK2)
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("Best α (1 ring)"); ax.set_ylim(-0.05, 1.05)
ax.set_title("f   Optimal smoothing", loc="left")

# g: patchiness of the predicted domains vs alpha
ax = fig.add_subplot(gs[2, 0:4]); gridy(ax)
for ds in ["DLPFC", "prostate"]:
    w = ring_table(ds, 1, "pas")
    ax.plot(w.columns, w.mean(), color=DSCOL[ds], lw=1.4, marker="o", ms=2.4, mec=SURF, mew=0.4)
    ax.axhline(LP.loc[LP["dataset"] == ds, "label_pas"].mean(), color=DSCOL[ds], lw=0.8, ls="--")
ax.text(0.98, 0.97, "dashed: patchiness of\nthe reference labels", transform=ax.transAxes, ha="right", va="top", fontsize=4.9, color=INK2)
ax.set_xlabel("Neighbour weight α (1 ring)"); ax.set_ylabel("PAS of predicted domains")
ax.set_title("g   Smoothing removes patches", loc="left")

# h, i: BANKSY lambda
b = SD[SD["method"].str.startswith("banksy-lam")].copy(); b["lev"] = b["method"].str[10:].astype(float)
ax = fig.add_subplot(gs[2, 4:8]); gridy(ax); zero(ax)
for ds in ["DLPFC", "prostate"]:
    w = b[b["dataset"] == ds].groupby(["slide", "lev"])["ari"].mean().unstack()
    curve(ax, w, 0.0, DSCOL[ds])
ax.axvline(0.2, color=MUTED, lw=0.6, ls=":"); ax.text(0.21, 0.02, "benchmark\nsetting", transform=ax.get_xaxis_transform(), fontsize=4.8, color=INK2)
ax.set_xlabel("BANKSY spatial weight λ"); ax.set_ylabel("ΔARI vs λ = 0"); ax.set_title("h   BANKSY: spatial weight", loc="left")
ax = fig.add_subplot(gs[2, 8:12])
w = b[b["dataset"].isin(["DLPFC", "prostate"])].groupby(["slide", "lev"])["ari"].mean().unstack()
dl = w.sub(w[0.0], axis=0).drop(columns=0.0)
dl = dl.loc[sorted(dl.index, key=lambda s: ("Erickson" in s, GS.loc[s, "boundary_frac"]))]
divheat(ax, dl.values, 0.3, [short(s) for s in dl.index], [f"{a:.1f}".replace("0.", ".") for a in dl.columns], fs=3.6)
ax.tick_params(axis="y", labelsize=4.4); ax.yaxis.tick_right()
nd = sum("Maynard" in s for s in dl.index); ax.axhline(nd - 0.5, color=SURF, lw=1.5)
ax.set_xlabel("λ"); ax.set_title("i   BANKSY: ΔARI per section", loc="left")

# j: STAGATE radius
ax = fig.add_subplot(gs[3, 0:4]); gridy(ax); zero(ax)
st = SD[SD["method"].str.startswith("stagate-r")].copy(); st["f"] = st["method"].str[9:].astype(float)
st0 = S[(S["method"] == "stagate") & S["slide"].isin(st["slide"].unique()) & S["seed"].isin([0, 1, 2])].assign(f=1.1)
st = pd.concat([st0, st]); st["nb"] = st["f"].map({1.1: 6, 2.1: 18, 3.1: 36})
for ds in ["DLPFC", "prostate"]:
    w = st[st["dataset"] == ds].groupby(["slide", "nb"])["ari"].mean().unstack()
    curve(ax, w, 6, DSCOL[ds])
ax.set_xticks([6, 18, 36]); ax.set_xlabel("STAGATE graph neighbours per spot"); ax.set_ylabel("ΔARI vs 6 neighbours")
ax.set_title("j   STAGATE: graph radius", loc="left")

# k, l: prostate tumour F1 and coarse ARI
for c0, met, lt, tt, yl in [(4, "tumor_f1", "k", "Prostate tumour F1", "ΔF1 vs α = 0"), (8, "ari_coarse", "l", "Prostate coarse ARI", "Δcoarse ARI vs α = 0")]:
    ax = fig.add_subplot(gs[3, c0:c0 + 4]); gridy(ax); zero(ax)
    for ring, ls in [(1, "-"), (2, "--"), (3, ":")]:
        curve(ax, ring_table("prostate", ring, met), 0.0, VIOLET, ls=ls, band=ring == 1, lw=1.3 if ring == 1 else 1.0)
    ax.set_xlabel("Neighbour weight α"); ax.set_ylabel(yl); ax.set_title(f"{lt}   {tt}", loc="left")
    if met == "tumor_f1":
        ax.legend(handles=[Line2D([], [], color=VIOLET, ls=ls, lw=1.1, label=f"{r} ring{'s' if r > 1 else ''}") for r, ls in [(1, "-"), (2, "--"), (3, ":")]],
                  loc="lower left", fontsize=5, handlelength=2.2)

# m-o: one prostate section without and with strong smoothing
sid = next(s for s in GS.index if s.endswith("V1_2"))
a = bu.load_slide(sid); lab = bu.scored_labels(a)
Z = np.asarray(bu.standard_pca(a, seed=0).obsm["X_pca"], float); xy = np.asarray(a.obsm["spatial"], float)
nb, _ = bu.lattice_neighbors(xy, 1.1)
r_i = np.concatenate([np.full(len(v), i) for i, v in enumerate(nb)]); c_i = np.concatenate(nb).astype(int)
Wm = sp.csr_matrix((np.ones(len(r_i)), (r_i, c_i)), shape=(len(nb), len(nb))); deg = np.asarray(Wm.sum(1)).ravel()
WZ = np.asarray(sp.diags(1 / np.maximum(deg, 1)) @ Wm @ Z); WZ[deg == 0] = Z[deg == 0]
k = bu.choose_k(a); labv = lab.values.astype(object); ok = pd.notna(labv)
cls = [c for c in lab.value_counts().index if lab.value_counts()[c] >= 20]
maps = []
for i, (alpha, tt) in enumerate([(None, "annotation"), (0.0, "GMM (α = 0)"), (1.0, "PCA-smooth, α = 1")]):
    ax = fig.add_subplot(gs[4, i * 4:(i + 1) * 4]); maps.append(ax)
    if alpha is None:
        spatial(ax, xy, label_colors(labv), fig); direct_labels(ax, xy, labv, cls, fs=5)
        ax.set_title(f"m   Prostate {short(sid)}: {tt}", loc="left", fontsize=6.4)
    else:
        pred = bu.gmm_tied((1 - alpha) * Z + alpha * WZ, k, 0)
        ari = adjusted_rand_score(labv[ok].astype(str), pred[ok])
        spatial(ax, xy, match_colors(pred, labv, cls), fig)
        ax.set_title(f"{'no'[i - 1]}   {tt}: ARI {ari:.2f}", loc="left", fontsize=6.4, fontweight="normal")
legend_below(fig, maps, class_handles(cls), gap=0.006, fs=5.4)
dataset_legend(fig, ("DLPFC", "prostate"), anchor=(0.5, 0.905), loc="lower center", n=NSEC, fontsize=5.8)
save(fig, "Fig6_dose_response")
print("FIG6_DONE")
