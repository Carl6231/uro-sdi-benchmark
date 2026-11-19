"""54_fig4_tls.py - Figure 4: tertiary lymphoid structure (TLS) detection in clear-cell RCC (panels a-o)."""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

S = scores(); main = S[S["method"].isin(METHODS)]
T = main[main["tls_f1"].notna()]
BT = pd.read_csv(os.path.join(TAB, "bootstrap_vs_gmm.csv"))
GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
SD = scores("results/scores_dose.csv")
rng = np.random.default_rng(0)
F = T.groupby(["method", "slide"])["tls_f1"].mean().unstack()            # mean over K = 6 and 10
order = F.mean(1).sort_values(ascending=False).index.tolist()
Y = {m: len(order) - 1 - i for i, m in enumerate(order)}
ntls = GC[GC["cls"] == "TLS"].set_index("slide")
sid = ntls.loc[ntls.index.isin(F.columns), "n_spots"].idxmax()
TP, FP, FN, TN = VIOLET, ORANGE, "#c4bdf3", "#efeee9"

fig = plt.figure(figsize=(7.4, 10.4))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.45, 1.45, 1.45, 1.45, 1.3], hspace=0.62, wspace=2.6)

# a-f: annotation and the best-matching TLS domain of five methods (K = 6, seed 0)
lab, xy_all = load_section(sid)
show = ["gmm", "smooth", "stagate", "sedr", "graphst"]
spots = run_labels(sid, "gmm", 6, 0).index
xy = xy_all.loc[spots].values; labs = lab.reindex(spots).values
pos = labs == "TLS"; neg = labs == "NO_TLS"
maps = []
for i, m in enumerate([None] + show):
    ax = fig.add_subplot(gs[i // 3, (i % 3) * 4:(i % 3) * 4 + 4]); maps.append(ax)
    if m is None:
        spatial(ax, xy, label_colors(labs), fig)
        ax.set_title(f"a   {short(sid)}: {int(pos.sum())} TLS spots", loc="left")
        continue
    cl = run_labels(sid, m, 6, 0).reindex(spots).values
    best, bf = None, -1
    for c in np.unique(cl):
        tp = np.sum((cl == c) & pos); fp = np.sum((cl == c) & neg); fn = np.sum((cl != c) & pos)
        f = 2 * tp / (2 * tp + fp + fn) if tp else 0
        if f > bf: best, bf = c, f
    inc = cl == best
    col = np.where(pos & inc, TP, np.where(neg & inc, FP, np.where(pos & ~inc, FN, np.where(neg, TN, UNLABELLED))))
    spatial(ax, xy, col, fig)
    ax.set_title(f"{'bcdef'[i - 1]}   {NAME[m]}: F1 {bf:.2f}", loc="left", fontweight="normal")
legend_below(fig, maps[3:], [Patch(color=c, label=l) for c, l in [(TP, "TLS, in the TLS domain"), (FN, "TLS, missed"),
                                                                 (FP, "not TLS, in the TLS domain"), (TN, "not TLS")]], gap=0.004)

# g: F1 per method; h: difference from GMM; i: precision vs recall
ax = fig.add_subplot(gs[2, 0:4]); gridx(ax)
strip(ax, {m: F.loc[m].values for m in order}, order, rng, xlab="TLS F1 (best-matching domain)")
allpos = T.drop_duplicates("slide")["tls_f1_all_positive"].mean()
ax.axvline(allpos, color=MUTED, lw=0.7, ls=":")
ax.set_title(f"g   TLS F1 ({F.shape[1]} sections)", loc="left")
ax = fig.add_subplot(gs[2, 4:8]); gridx(ax); zero(ax, "x")
b = BT[(BT["dataset"] == "ccRCC") & (BT["metric"] == "tls_f1")].set_index("method")
for m in order:
    if m == "gmm":
        ax.text(0.003, Y[m], "reference", fontsize=4.8, color=INK2, va="center"); continue
    r = b.loc[m]
    ax.plot([r["ci95_lo"], r["ci95_hi"]], [Y[m]] * 2, color=COL[m], lw=1.1)
    ax.scatter(r["mean_diff"], Y[m], s=14, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3)
    ax.text(1.03, Y[m], f"{int(r['n_better'])}/{int(r['n_slides'])}", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=4.8, color=INK2, clip_on=False)
ax.text(1.03, len(order) - 0.3, "better", transform=ax.get_yaxis_transform(), ha="left", fontsize=4.8, color=INK2)
method_rows(ax, order, labels=False)
ax.set_xlabel("ΔF1 vs GMM (95% bootstrap CI)"); ax.set_title("h   Difference from GMM", loc="left")
ax = fig.add_subplot(gs[2, 9:12]); gridx(ax)
pr = T.groupby("method")[["tls_precision", "tls_recall"]].mean()
for m in order:
    p_, r_ = pr.loc[m, "tls_precision"], pr.loc[m, "tls_recall"]
    ax.plot([p_, r_], [Y[m]] * 2, color=GRID, lw=1.3, zorder=1)
    ax.scatter(p_, Y[m], s=12, color=COL[m], edgecolor=SURF, lw=0.5, zorder=3)
    ax.scatter(r_, Y[m], s=12, color=SURF, edgecolor=COL[m], lw=0.8, zorder=3)
method_rows(ax, order, labels=False)
ax.set_xlabel("Precision (filled), recall (open)"); ax.set_title("i   Precision and recall", loc="left")

# j: K = 6 vs 10; k: per-section F1; l: gain vs TLS compactness
ax = fig.add_subplot(gs[3, 0:3]); gridx(ax)
FK = T.groupby(["method", "k"])["tls_f1"].mean().unstack()
for m in order:
    ax.plot([FK.loc[m, 6], FK.loc[m, 10]], [Y[m]] * 2, color=GRID, lw=1.3, zorder=1)
    ax.scatter(FK.loc[m, 6], Y[m], s=12, color=COL[m], edgecolor=SURF, lw=0.5, zorder=3)
    ax.scatter(FK.loc[m, 10], Y[m], s=12, color=SURF, edgecolor=COL[m], lw=0.8, zorder=3)
method_rows(ax, order)
ax.set_xlabel("TLS F1 (filled K = 6, open K = 10)"); ax.set_title("j   K = 6 vs 10", loc="left")
ax = fig.add_subplot(gs[3, 3:8])
cols_k = F.loc["gmm"].sort_values(ascending=False).index
heat(ax, F.loc[order, cols_k].values, SEQ[:11], 0, 0.8, fmt="{:.2f}", fs=3.9, xt=[short(s)[-4:] for s in cols_k], yt=[])
ax.tick_params(axis="x", rotation=90, labelsize=4.8)
ax.set_title("k   TLS F1 per section (GSM id, last 4 digits)", loc="left")
ax = fig.add_subplot(gs[3, 9:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
gain = F.loc[[m for m in SP9 if m in F.index]].mean() - F.loc["gmm"]
bfr = ntls.loc[gain.index, "boundary_frac"]
ax.scatter(bfr, gain, s=13, color=VIOLET, edgecolor=SURF, lw=0.5, zorder=3)
r_ = spearmanr(bfr, gain)
ax.set_xlabel("TLS boundary share"); ax.set_ylabel(f"ΔF1 vs GMM (mean of {NSP})")
ax.set_title(f"l   Gain vs TLS shape (ρ = {r_[0]:.2f})", loc="left")

# m: F1 vs TLS prevalence; n: TLS geometry; o: smoothing dose
ax = fig.add_subplot(gs[4, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
prev = T.groupby("slide")["tls_prevalence"].first() * 100
for m, mk in [("gmm", "o"), ("stagate", "s"), ("smooth", "D")]:
    ax.scatter(prev[F.columns], F.loc[m], s=11, marker=mk, color=COL[m], edgecolor=SURF, lw=0.4, zorder=3, label=NAME[m])
ax.set_xscale("log"); ax.set_xlabel("TLS spots (% of labelled)"); ax.set_ylabel("TLS F1")
r_ = spearmanr(prev[F.columns], F.loc[order].mean())
ax.legend(loc="lower right", fontsize=5, handletextpad=0.1)
ax.set_title(f"m   Rarer TLS, lower F1 (ρ = {r_[0]:.2f})", loc="left")
ax = fig.add_subplot(gs[4, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
g = ntls.loc[ntls.index.isin(F.columns)]
ax.scatter(g["n_spots"], g["n_patches"] + rng.uniform(-0.08, 0.08, len(g)), s=13, color=VIOLET, edgecolor=SURF, lw=0.5, zorder=3)
ax.set_xscale("log"); ax.set_xlabel("TLS spots per section"); ax.set_ylabel("Separate TLS patches")
ax.set_yticks(range(1, int(g["n_patches"].max()) + 1))
ax.set_title(f"n   Few, compact TLS patches", loc="left")
ax = fig.add_subplot(gs[4, 8:12]); gridy(ax); zero(ax)
dd = D[(D["dataset"] == "ccRCC") & D["tls_f1"].notna()]
for ring, ls in [(1, "-"), (2, "--"), (3, ":")]:
    sub = pd.concat([dd[dd["alpha"] == 0].assign(ring=ring), dd[dd["ring"] == ring]])
    w = sub.groupby(["slide", "alpha"])["tls_f1"].mean().unstack(); dl = w.sub(w[0.0], axis=0)
    ax.plot(dl.columns, dl.mean(), color=INK, ls=ls, lw=1.1)
    if ring == 1: ax.text(1.02, dl.mean().iloc[-1], "PCA-smooth\n1-3 rings", fontsize=4.6, va="center", color=INK2)
bb_ = SD[SD["method"].str.startswith("banksy-lam") & SD["tls_f1"].notna()].copy(); bb_["lev"] = bb_["method"].str[10:].astype(float)
w = bb_.groupby(["slide", "lev"])["tls_f1"].mean().unstack(); dl = w.sub(w[0.0], axis=0)
ax.plot(dl.columns, dl.mean(), color=BLUE, lw=1.1, marker="o", ms=2.2)
ax.text(dl.columns[-1] + 0.05, dl.mean().iloc[-1], "BANKSY λ", fontsize=4.6, va="center", color=INK2)
ax.set_xlim(-0.03, 1.28); ax.set_xlabel("Smoothing weight (α or λ)"); ax.set_ylabel("ΔTLS F1 vs none")
ax.set_title("o   Smoothing helps TLS", loc="left")
fig.legend(handles=category_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=3, fontsize=5.8, handletextpad=0.2)
save(fig, "Fig4_tls_ccrcc")
print("FIG4_DONE")
