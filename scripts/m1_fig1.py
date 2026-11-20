"""m1_fig1.py - main Figure 1: design, data, and the reversal between cortex and prostate cancer (panels a-n)."""
from mstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

S = scores(); main = S[S["method"].isin(METHODS)]
BT = pd.read_csv(os.path.join(TAB, "bootstrap_vs_gmm.csv"))
secs = {ds: sorted(main.loc[main["dataset"] == ds, "slide"].unique()) for ds in ["DLPFC", "prostate", "ccRCC"]}
tls_secs = sorted(S.loc[S["tls_f1"].notna(), "slide"].unique())
n_other = main.loc[main["dataset"] == "other", "slide"].nunique()
rng = np.random.default_rng(0)


def per(ds, met):
    return main[(main["dataset"] == ds) & main[met].notna()].groupby(["method", "slide"])[met].mean().unstack()


P = {ds: per(ds, "ari") for ds in ["DLPFC", "prostate"]}
order = P["DLPFC"].mean(1).sort_values(ascending=False).index.tolist()
Y = {m: len(order) - 1 - i for i, m in enumerate(order)}
R = pd.DataFrame({ds: P[ds].mean(1) for ds in P})

fig = plt.figure(figsize=(FW, 9.45))
gs = GridSpec(5, 12, figure=fig, height_ratios=[0.7, 1.55, 1.55, 1.45, 1.3], hspace=0.55, wspace=2.2)

# a: study design
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
boxes = [("Public Visium data", f"{len(secs['DLPFC'])} DLPFC · {len(secs['prostate'])} prostate\n{len(secs['ccRCC'])} ccRCC · "
          f"{n_other} unlabelled\n20 breast · 3 Zurich"),
         ("13 methods", "10 published spatial\nPCA-smooth\nGMM, Leiden"),
         ("Same K, same seeds", "K = annotated classes\n1–5 seeds per section\n8,480 runs"),
         ("Accuracy", "ARI, NMI, coarse ARI\ntumour and TLS F1\nclass-level F1"),
         ("When and why", "label grain, voting\nsmoothing dose\nsimulation, pair errors"),
         ("Checks", "breast, Zurich TLS\nmarker programmes\nK, seeds, run time")]
w, gap = 0.157, 0.0115
for i, (h, t) in enumerate(boxes):
    x0 = 0.003 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, 0.02), w, 0.86, boxstyle="round,pad=0.006,rounding_size=0.02", fc="#f3f2ee", ec=AXIS,
                                lw=0.7, clip_on=False))
    ax.text(x0 + w / 2, 0.73, h, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(x0 + w / 2, 0.33, t, ha="center", va="center", fontsize=6.0 if i else 5.8, color=INK2, linespacing=1.3)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x0 + w + 0.001, 0.45), (x0 + w + gap - 0.001, 0.45), arrowstyle="-|>",
                                     mutation_scale=6, color=INK2, lw=0.7))
ptitle(ax, "a", "Study design")

# b-d: reference annotations of the three labelled settings
I = pd.read_csv(os.path.join(TAB, "section_overview.csv")).set_index("slide")
tls_best = I.loc[tls_secs, "n_tls"].idxmax()
for col, (sid, lt, txt) in enumerate([(next(s for s in secs["DLPFC"] if s.endswith("151673")), "b", "DLPFC 151673: layers"),
                                       (next(s for s in secs["prostate"] if s.endswith("H1_2")), "c", "Prostate H1_2: pathology"),
                                       (tls_best, "d", f"ccRCC {short(tls_best)}: TLS")]):
    ax = fig.add_subplot(gs[1, col * 4:(col + 1) * 4])
    lab, xy = load_section(sid)
    spatial(ax, xy.values, label_colors(lab.values), fig)
    vc = lab.value_counts()
    direct_labels(ax, xy.values, lab.values, [c for c in vc.index if vc[c] >= 20], fs=6.3)
    ptitle(ax, lt, txt)

# e, f: ARI per section and method; g: difference from GMM
for c0, ds, lt in [(0, "DLPFC", "e"), (4, "prostate", "f")]:
    ax = fig.add_subplot(gs[2, c0 + (1 if c0 == 0 else 0):c0 + 4]); gridx(ax)
    strip_m(ax, {m: P[ds].loc[m].values for m in order}, order, rng, xlab="ARI", labels=c0 == 0)
    ax.axvline(P[ds].loc["gmm"].mean(), color=MUTED, lw=0.7, ls=":", zorder=1)
    ptitle(ax, lt, f"{DSNAME[ds]} ({P[ds].shape[1]} sections)")
ax = fig.add_subplot(gs[2, 8:12]); gridx(ax); zero(ax, "x")
for ds, off in [("DLPFC", 0.17), ("prostate", -0.17)]:
    b = BT[(BT["dataset"] == ds) & (BT["metric"] == "ari")].set_index("method")
    for m in order:
        if m == "gmm":
            continue
        r = b.loc[m]; y = Y[m] + off
        ax.plot([r["ci95_lo"], r["ci95_hi"]], [y, y], color=DSCOL[ds], lw=1.0)
        ax.scatter(r["mean_diff"], y, s=13, color=DSCOL[ds] if r["q_bh"] < 0.05 else SURF, edgecolor=DSCOL[ds], lw=0.8, zorder=3)
ax.text(0.006, Y["gmm"], "GMM (reference)", fontsize=FMIN, color=INK2, va="center")
method_rows_m(ax, order, labels=False)
ax.set_xlabel("ΔARI vs GMM (95% CI)")
ptitle(ax, "g", "Difference from GMM")

# h: means in two tissues; i: rank change; j: three prostate readouts
ax = fig.add_subplot(gs[3, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
rho, p = spearmanr(R["DLPFC"], R["prostate"])
for m, r in R.iterrows():
    ax.scatter(r["DLPFC"], r["prostate"], s=26, color=COL[m], edgecolor=SURF, lw=0.8, zorder=3)
ax.set_xlim(R["DLPFC"].min() - 0.06, R["DLPFC"].max() + 0.08); ax.set_ylim(R["prostate"].min() - 0.045, R["prostate"].max() + 0.045)
ax.set_xlabel("Mean ARI, DLPFC"); ax.set_ylabel("Mean ARI, prostate")
ptitle(ax, "h", f"Two tissues (ρ = {rho:.2f}, P = {p:.2f})")
LAB = ["gmm", "smooth", "graphst", "stagate", "bass", "banksy", "spaceflow", "sedr"]      # the rest: colour and panel i
oo = LAB + [m for m in R.index if m not in LAB]
place_labels(ax, fig, R.loc[oo, "DLPFC"].values, R.loc[oo, "prostate"].values, [NAME[m] if m in LAB else "" for m in oo], fs=FMIN)

ax = fig.add_subplot(gs[3, 4:8])
rk1 = R["DLPFC"].rank(ascending=False); rk2 = R["prostate"].rank(ascending=False); n = len(R)
for m in R.index:
    hi = m in ("gmm", "smooth", "bass", "banksy")
    ax.plot([0.3, 0.7], [rk1[m], rk2[m]], color=COL[m], lw=1.4 if hi else 0.8, alpha=1 if hi else 0.55, zorder=3 if hi else 2)
    ax.scatter([0.3, 0.7], [rk1[m], rk2[m]], s=12, color=COL[m], edgecolor=SURF, lw=0.6, zorder=4)
    ax.text(0.27, rk1[m], NAME[m], ha="right", va="center", fontsize=FMIN, color=INK if hi else INK2)
    ax.text(0.73, rk2[m], NAME[m], ha="left", va="center", fontsize=FMIN, color=INK if hi else INK2)
ax.set_ylim(n + 0.6, 0.4); ax.set_xlim(0, 1); ax.set_xticks([0.3, 0.7]); ax.set_xticklabels(["DLPFC", "Prostate"])
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ["left", "bottom"]]
ptitle(ax, "i", "Rank (1 = best)")

ax = fig.add_subplot(gs[3, 9:12]); gridx(ax)
mets = [("ari", "ARI"), ("ari_coarse", "Coarse ARI"), ("tumor_f1", "Tumour F1")]
for i, (met, nm) in enumerate(mets):
    Q = per("prostate", met).mean(1); y = len(mets) - 1 - i
    sp = Q[[m for m in SP9 if m in Q]]
    ax.scatter(sp.values, y + rng.uniform(-0.12, 0.12, len(sp)), s=10, color=C_SPATIAL, alpha=0.45, lw=0, zorder=2)
    ax.scatter(sp.mean(), y + 0.28, s=24, marker="v", color=C_SPATIAL, edgecolor=SURF, lw=0.5, zorder=3)
    ax.scatter(Q["smooth"], y, s=22, color=C_SIMPLE, edgecolor=SURF, lw=0.6, zorder=4)
    ax.scatter(Q["gmm"], y, s=26, color=C_NONSP, edgecolor=SURF, lw=0.6, zorder=5)
ax.set_yticks(range(len(mets))); ax.set_yticklabels([nm for _, nm in mets][::-1]); ax.set_ylim(-0.6, len(mets) - 0.3)
ax.set_xlabel("Mean over 7 sections")
ax.text(0.02, 0.02, "▼ mean of 11 spatial", transform=ax.transAxes, ha="left", va="bottom", fontsize=FMIN, color=C_SPATIAL)
ptitle(ax, "j", "Prostate readouts")

# k-n: domains on prostate H1_2 (a typical seed of each method)
sid = next(s for s in secs["prostate"] if s.endswith("H1_2"))
lab, xy_all = load_section(sid)
CLS = ["Benign", "GG1", "Stroma"]
runs = main[main["slide"] == sid]
maps = []
for i, (m, lt) in enumerate(zip(["gmm", "smooth", "stagate", "bass"], "klmn")):
    seed, k, mean_ari = pick_seed(runs, m)
    cl = run_labels(sid, m, k, seed)
    spots = cl.index; xy = xy_all.loc[spots].values; labs = lab.reindex(spots)
    ax = fig.add_subplot(gs[4, i * 3:(i + 1) * 3]); maps.append(ax)
    spatial(ax, xy, match_colors(cl.values, labs.values, CLS), fig)
    ptitle(ax, lt, f"{NAME[m]}  ·  ARI {mean_ari:.2f}", fontsize=FS)
maps_legend(fig, maps, cls_handles(CLS, [(MUTED, "unmatched cluster")]) +
            [Line2D([], [], ls="", label="domains on prostate H1_2, coloured by the class they match")], gap=0.004)
top_legend(fig, cat_handles(("DLPFC", "prostate")), y=0.925)
msave(fig, "Figure1")
print("M1_DONE")
