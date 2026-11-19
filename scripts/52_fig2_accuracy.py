"""52_fig2_accuracy.py - Figure 2: accuracy of 11 methods on DLPFC and prostate cancer (panels a-o).
All method-row panels share one method order (best mean ARI on DLPFC at the top)."""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

S = scores(); main = S[S["method"].isin(METHODS)]
BT = pd.read_csv(os.path.join(TAB, "bootstrap_vs_gmm.csv"))
LP = pd.read_csv(os.path.join(TAB, "gain_vs_label_fragmentation.csv"))
CF = pd.read_csv(os.path.join(TAB, "class_f1.csv"))
rng = np.random.default_rng(0)


def per(ds, met):
    return main[(main["dataset"] == ds) & main[met].notna()].groupby(["method", "slide"])[met].mean().unstack()


P = {ds: per(ds, "ari") for ds in ["DLPFC", "prostate"]}
order = P["DLPFC"].mean(1).sort_values(ascending=False).index.tolist()
Y = {m: len(order) - 1 - i for i, m in enumerate(order)}
fig = plt.figure(figsize=(7.4, 10.9))
gs = GridSpec(6, 12, figure=fig, height_ratios=[1.45, 1.3, 1.45, 1.45, 1.35, 1.0], hspace=0.62, wspace=2.6)

# a, b: ARI per section; c: difference from GMM
for c0, ds, lt in [(0, "DLPFC", "a"), (4, "prostate", "b")]:
    ax = fig.add_subplot(gs[0, c0:c0 + 4]); gridx(ax)
    strip(ax, {m: P[ds].loc[m].values for m in order}, order, rng, xlab="ARI vs manual annotation", labels=c0 == 0)
    ax.axvline(P[ds].loc["gmm"].mean(), color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_title(f"{lt}   {DSNAME[ds]}: ARI ({P[ds].shape[1]} sections)", loc="left")
ax = fig.add_subplot(gs[0, 8:12]); gridx(ax); zero(ax, "x")
for ds, off in [("DLPFC", 0.17), ("prostate", -0.17)]:
    b = BT[(BT["dataset"] == ds) & (BT["metric"] == "ari")].set_index("method")
    for m in order:
        if m == "gmm":
            continue
        r = b.loc[m]; y = Y[m] + off
        ax.plot([r["ci95_lo"], r["ci95_hi"]], [y, y], color=DSCOL[ds], lw=1.0)
        ax.scatter(r["mean_diff"], y, s=12, color=DSCOL[ds] if r["q_bh"] < 0.05 else SURF, edgecolor=DSCOL[ds], lw=0.8, zorder=3)
ax.text(0.003, Y["gmm"], "reference", fontsize=4.8, color=INK2, va="center")
method_rows(ax, order, labels=False)
ax.set_xlabel("ΔARI vs GMM (95% bootstrap CI)")
ax.legend(handles=[Line2D([], [], color=INK2, marker="o", ms=3.2, lw=0.9, label="q < 0.05"),
                   Line2D([], [], color=INK2, marker="o", mfc=SURF, ms=3.2, lw=0.9, label="q ≥ 0.05")],
          loc="lower right", ncol=1, fontsize=5)
ax.set_title("c   Difference from GMM", loc="left")

# d, e: ARI per section and method
for c0, c1, ds, lt in [(0, 7, "DLPFC", "d"), (7, 12, "prostate", "e")]:
    ax = fig.add_subplot(gs[1, c0:c1])
    M = P[ds].loc[order]
    heat(ax, M.values, SEQ, 0, 0.8 if ds == "DLPFC" else 0.55, fmt="{:.2f}", fs=4.3, xt=[short(s) for s in M.columns],
         yt=[NAME[m] for m in order] if ds == "DLPFC" else [])
    ax.tick_params(axis="x", rotation=90, labelsize=5); ax.tick_params(axis="y", labelsize=5.4)
    ax.set_title(f"{lt}   {DSNAME[ds]}: ARI per section", loc="left")

# f: NMI; g: coarse ARI; h: tumour F1 (prostate)
ax = fig.add_subplot(gs[2, 0:4]); gridx(ax)
N = {ds: per(ds, "nmi").mean(1) for ds in ["DLPFC", "prostate"]}
for m in order:
    ax.plot([N["prostate"][m], N["DLPFC"][m]], [Y[m]] * 2, color=GRID, lw=1.5, zorder=1)
    for ds in ["DLPFC", "prostate"]:
        ax.scatter(N[ds][m], Y[m], s=14, color=DSCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
method_rows(ax, order); ax.set_xlabel("NMI (mean over sections)"); ax.set_title("f   NMI in both tissues", loc="left")
for c0, met, lt, tt, xl in [(4, "ari_coarse", "g", "Prostate: 3 merged classes", "Coarse ARI (tumour/benign/stroma)"),
                            (8, "tumor_f1", "h", "Prostate: tumour domain", "Tumour F1 (best-matching domain)")]:
    ax = fig.add_subplot(gs[2, c0:c0 + 4]); gridx(ax)
    Q = per("prostate", met)
    strip(ax, {m: Q.loc[m].values for m in order}, order, rng, xlab=xl, labels=False)
    ax.axvline(Q.loc["gmm"].mean(), color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_title(f"{lt}   {tt}", loc="left")

# i: seed variability; j: smoothness of the domains; k: per-class F1 (prostate)
ax = fig.add_subplot(gs[3, 0:4]); gridx(ax); row3 = [ax]
for ds in ["DLPFC", "prostate"]:
    sd = main[main["dataset"] == ds].groupby(["method", "slide"])["ari"].std().groupby("method").mean()
    ax.scatter([sd[m] for m in order], [Y[m] for m in order], s=14, color=DSCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
method_rows(ax, order); ax.set_xlabel("SD of ARI over 5 seeds"); ax.set_title("i   Seed-to-seed variability", loc="left")
ax = fig.add_subplot(gs[3, 4:8]); gridx(ax); row3.append(ax)
for ds in ["DLPFC", "prostate"]:
    pa = main[main["dataset"] == ds].groupby("method")["pas"].mean()
    ax.scatter([pa[m] for m in order], [Y[m] for m in order], s=14, color=DSCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
    ax.axvline(LP.loc[LP["dataset"] == ds, "label_pas"].mean(), color=DSCOL[ds], lw=0.8, ls="--", zorder=1)
method_rows(ax, order, labels=False)
ax.set_xlabel("PAS of the domains (dashed: labels)"); ax.set_title("j   Patchiness of the domains", loc="left")
ax = fig.add_subplot(gs[3, 8:12]); row3.append(ax)
cf = CF[CF["dataset"] == "prostate"].copy()
cf["grp"] = cf["cls"].map(lambda c: "Benign" if c.startswith("Benign") else ("Stroma" if c == "Stroma" else ("Tumour" if c.startswith("GG") else None)))
G = cf.dropna(subset=["grp"]).groupby(["method", "grp", "slide"])["f1"].mean().groupby(["method", "grp"]).mean().unstack()
heat(ax, G.loc[order, ["Benign", "Stroma", "Tumour"]].values, SEQ[:11], 0.2, 0.9, fmt="{:.2f}", fs=4.6, xt=["Benign", "Stroma", "Tumour"], yt=[])
ax.xaxis.tick_top(); ax.tick_params(axis="x", labelsize=5.4)
ax.set_title("k   Prostate: F1 per class", loc="left", pad=3)

# l: homogeneity vs completeness; m: accuracy vs patchiness; n: accuracy vs runtime
ax = fig.add_subplot(gs[4, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
hc = main[main["dataset"].isin(["DLPFC", "prostate"])].groupby(["dataset", "method"])[["homogeneity", "completeness"]].mean()
for (ds, m), r in hc.iterrows():
    ax.scatter(r["completeness"], r["homogeneity"], s=16, color=COL[m], marker="o" if ds == "DLPFC" else "s", edgecolor=SURF, lw=0.5, zorder=3)
lim = [hc.values.min() - 0.03, hc.values.max() + 0.03]
ax.plot(lim, lim, color=MUTED, lw=0.6, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.text(0.04, 0.96, "circles: DLPFC\nsquares: prostate", transform=ax.transAxes, va="top", fontsize=4.9, color=INK2)
ax.set_xlabel("Completeness"); ax.set_ylabel("Homogeneity"); ax.set_title("l   Homogeneity vs completeness", loc="left")
ax = fig.add_subplot(gs[4, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
txt = []
for ds in ["DLPFC", "prostate"]:
    d = main[main["dataset"] == ds].groupby(["method", "slide"])[["pas", "ari"]].mean()
    ax.scatter(d["pas"], d["ari"], s=5, color=DSCOL[ds], alpha=0.55, lw=0)
    txt.append(f"{DSNAME[ds]} ρ = {spearmanr(d['pas'], d['ari'])[0]:.2f}")
ax.text(0.97, 0.97, "\n".join(txt), transform=ax.transAxes, ha="right", va="top", fontsize=4.9, color=INK2)
ax.set_xlabel("PAS of the domains"); ax.set_ylabel("ARI"); ax.set_title("m   Accuracy vs patchiness", loc="left")
ax = fig.add_subplot(gs[4, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
rt = main.groupby("method")["sec"].median()
for m in order:
    ax.plot([rt[m]] * 2, [P["prostate"].loc[m].mean(), P["DLPFC"].loc[m].mean()], color=GRID, lw=1.2, zorder=1)
    for ds in ["DLPFC", "prostate"]:
        ax.scatter(rt[m], P[ds].loc[m].mean(), s=13, color=DSCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
ax.set_xscale("log"); ax.set_ylim(0.1, 0.6)
sel = ["gmm", "smooth", "graphst", "sedr", "bayesspace"]
place_labels(ax, fig, [rt[m] for m in sel], [P["DLPFC"].loc[m].mean() for m in sel], [NAME[m] for m in sel], fs=4.8)
ax.set_xlabel("Median seconds per run"); ax.set_ylabel("Mean ARI"); ax.set_title("n   Accuracy vs runtime", loc="left")

# o: ranks across metrics
ax = fig.add_subplot(gs[5, :])
cols_o = [("ARI DLPFC", P["DLPFC"].mean(1), False), ("ARI prostate", P["prostate"].mean(1), False), ("NMI DLPFC", N["DLPFC"], False),
          ("NMI prostate", N["prostate"], False), ("coarse ARI prostate", per("prostate", "ari_coarse").mean(1), False),
          ("tumour F1 prostate", per("prostate", "tumor_f1").mean(1), False),
          ("TLS F1 ccRCC", main[main["tls_f1"].notna()].groupby("method")["tls_f1"].mean(), False), ("runtime", rt, True)]
Rk = pd.DataFrame({l: s.rank(ascending=low, method="min") for l, s, low in cols_o}).loc[order].T
V = 12 - Rk.values
heat(ax, V, SEQ[2:], 1, 11, text=False, xt=[NAME[m] for m in order], yt=Rk.index.tolist())
for i in range(Rk.shape[0]):
    for j in range(Rk.shape[1]):
        ax.text(j, i, f"{int(Rk.values[i, j])}", ha="center", va="center", fontsize=5, color=SURF if V[i, j] >= 7 else INK)
ax.xaxis.tick_top(); ax.tick_params(axis="x", labelsize=5.6)
ax.set_title("o   Rank across metrics (1 = best)", loc="left", pad=12)
fig.legend(handles=category_handles(("DLPFC", "prostate")), loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=5, fontsize=5.6,
           handletextpad=0.2, columnspacing=1.0)
align_titles(fig, row3)
save(fig, "Fig2_accuracy")
print("FIG2_DONE")
