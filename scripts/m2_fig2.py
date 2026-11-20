"""m2_fig2.py - main Figure 2: TLS detection in clear-cell RCC, Meylan cohort and the independent Zurich cohort (a-q)."""
from mstyle import *
from scipy.stats import spearmanr
from matplotlib.patches import Patch

S = scores(); main = S[S["method"].isin(METHODS)]
T = main[main["tls_f1"].notna()]
BT = pd.read_csv(os.path.join(TAB, "bootstrap_vs_gmm.csv"))
GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
SD = scores("results/scores_dose.csv")
ZD = os.path.join(bu.BASE, "results/zurich")
if not os.path.exists(os.path.join(ZD, "zurich_tls_runs.csv")):        # a clone of the repository keeps them here
    ZD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external_validation")
ZR = pd.read_csv(os.path.join(ZD, "zurich_tls_runs.csv")); ZS = pd.read_csv(os.path.join(ZD, "zurich_tls_shuffled_neighbours.csv"))
ZP = pd.read_csv(os.path.join(ZD, "zurich_tls_patch_hits.csv")); ZC = pd.read_csv(os.path.join(ZD, "zurich_tls_spots.csv.gz"))
MY = pd.read_csv(os.path.join(ZD, "meylan_tls_sections.csv"))
rng = np.random.default_rng(0)
F = T.groupby(["method", "slide"])["tls_f1"].mean().unstack()            # mean over K = 6 and 10
order = F.mean(1).sort_values(ascending=False).index.tolist()
Y = {m: len(order) - 1 - i for i, m in enumerate(order)}
ntls = GC[GC["cls"] == "TLS"].set_index("slide")
sid = ntls.loc[ntls.index.isin(F.columns), "n_spots"].idxmax()
TP, FN, FP, TN = VIOLET, "#c4bdf3", ORANGE, "#e4e2da"
ACOL = {0.0: C_NONSP, 0.3: "#8fd6bb", 0.7: C_SIMPLE}          # alpha 0 is GMM; PCA-smooth weights in the baseline's colour
ANAME = {0.0: "α = 0 (GMM)", 0.3: "α = 0.3", 0.7: "α = 0.7"}

fig = plt.figure(figsize=(FW, 9.4))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.2, 1.55, 1.3, 1.2, 1.3], hspace=0.62, wspace=2.2)

# a-d: Meylan section with the most TLS spots; the TLS domain of three methods (K = 6, seed 0)
lab, xy_all = load_section(sid)
spots = run_labels(sid, "gmm", 6, 0).index
xy = xy_all.loc[spots].values; labs = lab.reindex(spots).values
pos = labs == "TLS"; neg = labs == "NO_TLS"
maps = []
for i, m in enumerate([None, "gmm", "smooth", "stagate"]):
    ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3]); maps.append(ax)
    if m is None:
        spatial(ax, xy, np.where(pos, TP, np.where(neg, TN, UNLABELLED)), fig)
        ptitle(ax, "a", f"Meylan {short(sid)}", fontsize=FS)
        continue
    cl = run_labels(sid, m, 6, 0).reindex(spots).values
    best, bf = None, -1
    for c in np.unique(cl):
        tp = np.sum((cl == c) & pos); fp = np.sum((cl == c) & neg); fn = np.sum((cl != c) & pos)
        f = 2 * tp / (2 * tp + fp + fn) if tp else 0
        if f > bf: best, bf = c, f
    inc = cl == best
    spatial(ax, xy, np.where(pos & inc, TP, np.where(neg & inc, FP, np.where(pos & ~inc, FN, np.where(neg, TN, UNLABELLED)))), fig)
    ptitle(ax, "bcd"[i - 1], f"{NAME[m]}  ·  F1 {bf:.2f}", fontsize=FS)
maps_legend(fig, maps, [Patch(color=c, label=l) for c, l in [(TP, "TLS, called"), (FN, "TLS, missed"),
                                                            (FP, "not TLS, called"), (TN, "not TLS")]], gap=0.004)

# e: TLS F1 per method; f: difference from GMM; g: precision and recall
ax = fig.add_subplot(gs[1, 1:4]); gridx(ax)
strip_m(ax, {m: F.loc[m].values for m in order}, order, rng, xlab="TLS F1")
ax.axvline(F.loc["gmm"].mean(), color=MUTED, lw=0.7, ls=":", zorder=1)
ptitle(ax, "e", f"TLS F1 ({F.shape[1]} sections)")
ax = fig.add_subplot(gs[1, 4:7]); gridx(ax); zero(ax, "x")
b = BT[(BT["dataset"] == "ccRCC") & (BT["metric"] == "tls_f1")].set_index("method")
for m in order:
    if m == "gmm":
        ax.text(0.004, Y[m], "reference", fontsize=FMIN, color=INK2, va="center"); continue
    r = b.loc[m]
    ax.plot([r["ci95_lo"], r["ci95_hi"]], [Y[m]] * 2, color=COL[m], lw=1.1)
    ax.scatter(r["mean_diff"], Y[m], s=15, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3)
    ax.text(1.04, Y[m], f"{int(r['n_better'])}/{int(r['n_slides'])}", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=FMIN, color=INK2, clip_on=False)
ax.text(1.04, len(order) - 0.35, "better", transform=ax.get_yaxis_transform(), ha="left", fontsize=FMIN, color=INK2)
method_rows_m(ax, order, labels=False)
ax.set_xlabel("ΔF1 vs GMM (95% CI)"); ptitle(ax, "f", "vs GMM")
ax = fig.add_subplot(gs[1, 8:12]); gridx(ax)
pr = T.groupby("method")[["tls_precision", "tls_recall"]].mean()
for m in order:
    p_, r_ = pr.loc[m, "tls_precision"], pr.loc[m, "tls_recall"]
    ax.plot([p_, r_], [Y[m]] * 2, color=GRID, lw=1.4, zorder=1)
    ax.scatter(p_, Y[m], s=14, color=COL[m], edgecolor=SURF, lw=0.5, zorder=3)
    ax.scatter(r_, Y[m], s=14, color=SURF, edgecolor=COL[m], lw=0.9, zorder=3)
method_rows_m(ax, order, labels=False)
ax.set_xlabel("Precision ● / recall ○"); ptitle(ax, "g", "Precision, recall")

# h: F1 vs TLS prevalence; i: TLS patch sizes in the two cohorts; j: smoothing dose
ax = fig.add_subplot(gs[2, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
prev = T.groupby("slide")["tls_prevalence"].first() * 100
for m, mk in [("gmm", "o"), ("stagate", "s"), ("smooth", "D")]:
    ax.scatter(prev[F.columns], F.loc[m], s=14, marker=mk, color=COL[m], edgecolor=SURF, lw=0.4, zorder=3, label=NAME[m])
ax.set_xscale("log"); ax.set_xlabel("TLS spots (% of labelled)"); ax.set_ylabel("TLS F1")
r_ = spearmanr(prev[F.columns], F.loc[order].mean())
ax.legend(loc="upper left", fontsize=FMIN, handletextpad=0.1, borderaxespad=0.2)
ptitle(ax, "h", f"Rarer TLS, lower F1 (ρ = {r_[0]:.2f})")

ax = fig.add_subplot(gs[2, 5:8]); gridy(ax)
my = MY[MY["eligible_original_TLS_F1"].astype(str) == "True"]
msz = np.concatenate([np.array(str(s).split(","), float) for s in my["patch_sizes_spots"].dropna()])
groups = [("Meylan", msz, VIOLET)] + [(s, ZP.loc[ZP["sample"] == s, "size_spots"].values, INK2) for s in ["KC1", "KC2", "KC3"]]
for i, (nm, v, c) in enumerate(groups):
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=c, alpha=0.6, lw=0, zorder=2)
    ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.2, zorder=3)
    ax.text(i + 0.32, np.median(v), f"{np.median(v):g}", ha="left", va="center", fontsize=FMIN, color=INK)
ax.set_yscale("log"); ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("Spots per TLS patch"); ax.set_xlim(-0.6, len(groups) - 0.4)
ptitle(ax, "i", "TLS patch size")

ax = fig.add_subplot(gs[2, 8:12]); gridy(ax); zero(ax)
dd = D[(D["dataset"] == "ccRCC") & D["tls_f1"].notna()]
for ring, ls in [(1, "-"), (2, "--"), (3, ":")]:
    sub = pd.concat([dd[dd["alpha"] == 0].assign(ring=ring), dd[dd["ring"] == ring]])
    w = sub.groupby(["slide", "alpha"])["tls_f1"].mean().unstack(); dl = w.sub(w[0.0], axis=0)
    ax.plot(dl.columns, dl.mean(), color=C_SIMPLE, ls=ls, lw=1.3, label=f"α, {ring} ring{'s' if ring > 1 else ''}")
bb_ = SD[SD["method"].str.startswith("banksy-lam") & SD["tls_f1"].notna()].copy(); bb_["lev"] = bb_["method"].str[10:].astype(float)
w = bb_.groupby(["slide", "lev"])["tls_f1"].mean().unstack(); dl = w.sub(w[0.0], axis=0)
ax.plot(dl.columns, dl.mean(), color=C_SPATIAL, lw=1.3, marker="o", ms=2.5, label="BANKSY λ")
ax.set_xlabel("Smoothing weight (α or λ)"); ax.set_ylabel("ΔTLS F1 vs none")
ax.set_ylim(-0.012, 0.2); ax.set_yticks([0, 0.05, 0.1])
ax.legend(loc="upper left", fontsize=FMIN, ncol=2, handlelength=1.8, columnspacing=0.8, borderaxespad=0.2,
          title="PCA-smooth (green), BANKSY (blue)", title_fontsize=FMIN)
ptitle(ax, "j", "Smoothing dose, Meylan")

# k-n: Zurich sections - reference TLS and the TLS cluster without and with smoothing (seed 0)
zmaps = []
for i, (smp, al, lt) in enumerate([("KC1", None, "k"), ("KC1", 0.0, "l"), ("KC1", 0.7, "m"), ("KC3", 0.7, "n")]):
    d = ZC[ZC["sample"] == smp]
    ref = d["manual_anno"].eq("TLS").to_numpy(); known = d["manual_anno"].isin(["TLS", "INFL", "NOR", "TUM"]).to_numpy()
    ax = fig.add_subplot(gs[3, i * 3:(i + 1) * 3]); zmaps.append(ax)
    if al is None:
        col = np.where(ref, TP, np.where(known, TN, UNLABELLED)); ttl = f"Zurich {smp}: reference TLS"
    else:
        call = d[f"call_a{al:.1f}_s0"].to_numpy(bool)
        col = np.where(ref & call, TP, np.where(ref, FN, np.where(known & call, FP, np.where(known, TN, UNLABELLED))))
        f1 = ZR[(ZR["sample"] == smp) & (ZR["alpha"] == al) & (ZR["seed"] == 0)]["tls_f1"].item()
        ttl = f"{smp}, {'GMM' if al == 0 else f'α = {al:g}'}  ·  F1 {f1:.2f}"
    spatial(ax, d[["x", "y"]].to_numpy(float), col, fig)
    ptitle(ax, lt, ttl, fontsize=FS)
maps_legend(fig, zmaps, [Patch(color=c, label=l) for c, l in [(TP, "TLS, called"), (FN, "TLS, missed"), (FP, "not TLS, called"),
                                                              (TN, "not TLS"), (UNLABELLED, "not annotated")]], gap=0.004)

# o: Zurich TLS F1 by smoothing weight with the shuffled-neighbour control; p: detection by patch size; q: precision vs recall
ax = fig.add_subplot(gs[4, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for smp, mk in [("KC1", "o"), ("KC2", "s"), ("KC3", "D")]:
    g = ZR[ZR["sample"] == smp].groupby("alpha")["tls_f1"]
    ax.plot(g.mean().index, g.mean().values, color=INK2, lw=1.0, marker=mk, ms=3.5, mfc=INK2, mec=SURF, zorder=3)
    for al, v in ZR[ZR["sample"] == smp].groupby("alpha")["tls_f1"]:
        ax.scatter(np.full(len(v), al), v, s=8, color=ACOL[al], lw=0, zorder=2, alpha=0.8)
    sh = ZS[ZS["sample"] == smp]["tls_f1"]
    ax.scatter(np.full(len(sh), 0.86) + rng.uniform(-0.03, 0.03, len(sh)), sh, s=8, marker=mk, color=SURF, edgecolor=MUTED, lw=0.6)
    ax.plot([0.81, 0.91], [sh.median()] * 2, color=MUTED, lw=1.0)
    ax.text(0.72, g.mean().loc[0.7], smp, fontsize=FMIN, va="center", ha="left", color=INK2)
ax.set_xticks([0, 0.3, 0.7, 0.86]); ax.set_xticklabels(["0", "0.3", "0.7", "0.7\nshuffled"]); ax.set_xlim(-0.06, 0.95)
ax.set_xlabel("Neighbour weight α"); ax.set_ylabel("TLS F1")
ptitle(ax, "o", "Zurich: F1 needs true neighbours")

ax = fig.add_subplot(gs[4, 5:8]); gridy(ax)
bins = [(1, 3, "1–3"), (4, 9, "4–9"), (10, 1e9, "≥ 10")]
for j, al in enumerate([0.0, 0.3, 0.7]):
    hit = np.stack([(ZP[f"hits_a{al:.1f}_s{s}"] > 0).astype(float) for s in (0, 1, 2)], 1).mean(1)
    vals = [hit[(ZP["size_spots"] >= lo) & (ZP["size_spots"] <= hi)].mean() * 100 for lo, hi, _ in bins]
    ax.bar(np.arange(len(bins)) + (j - 1) * 0.27, vals, width=0.25, color=ACOL[al], label=ANAME[al])
ns = [int(((ZP["size_spots"] >= lo) & (ZP["size_spots"] <= hi)).sum()) for lo, hi, _ in bins]
ax.set_xticks(range(len(bins))); ax.set_xticklabels([f"{nm}\n({n})" for (_, _, nm), n in zip(bins, ns)])
ax.set_ylim(0, 115); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("TLS patches detected (%)")
ax.set_xlabel("Patch size, spots (n patches)")
ax.legend(loc="upper left", fontsize=FMIN, ncol=3, handlelength=0.8, columnspacing=0.6, borderaxespad=0.1, handletextpad=0.3)
ptitle(ax, "p", "Small TLS are lost")

ax = fig.add_subplot(gs[4, 9:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
pr = ZR.groupby(["sample", "alpha"])[["recall", "precision"]].mean()
for smp in ["KC1", "KC2", "KC3"]:
    g = pr.loc[smp]
    ax.plot(g["recall"], g["precision"], color=GRID, lw=1.2, zorder=1)
    for al, r in g.iterrows():
        ax.scatter(r["recall"], r["precision"], s=22, color=ACOL[al], edgecolor=SURF, lw=0.5, zorder=3)
    ax.text(g["recall"].iloc[-1] + 0.03, g["precision"].iloc[-1], smp, fontsize=FMIN, va="center", color=INK2)
ax.set_xlim(0.25, 1.12); ax.set_ylim(0, 0.95); ax.set_yticks([0.2, 0.4, 0.6, 0.8]); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ptitle(ax, "q", "Precision rises")
top_legend(fig, cat_handles(), y=0.925)
msave(fig, "Figure2")
print("M2_DONE")
