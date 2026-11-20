"""83_fig11_guide.py - Figure 11: a practical guide (panels a-h).
a, b  scorecard of all methods over the whole study (labelled data, simulation, label-free markers, cost);
c-e   when does spatial information help? class-level model (gain ~ boundary share + log10 class size, random intercept per
      section) fitted on DLPFC + prostate classes and tested on breast-cancer classes;
f     how much smoothing (PCA-smooth dose) by grain of the labels; g  cost vs accuracy; h  decision guide.
Writes results/tables/guide_scorecard.csv and guide_breakeven.csv.
"""
from fig_common import *
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr, t as tdist
import statsmodels.formula.api as smf

rng = np.random.default_rng(11)
ALLM = list(METHODS)
SPM = [m for m in ALLM if m not in ("gmm", "leiden")]
DCOL = {"DLPFC": MUTED, "prostate": VIOLET, "breast": MAGENTA}
DNAME = {"DLPFC": "DLPFC", "prostate": "Prostate cancer", "breast": "Breast cancer (held out)"}


def rd(path):
    p = os.path.join(bu.BASE, path)
    return pd.read_csv(p) if os.path.exists(p) else None


def per_method(df, val):
    """mean over sections of the per-section mean over seeds"""
    return df.groupby(["slide", "method"])[val].mean().unstack().mean()


# ---------------- scorecard ----------------
S = scores(); S = S[S["method"].isin(ALLM)]
crit = {}
crit["DLPFC\nARI"] = per_method(S[S["dataset"] == "DLPFC"], "ari")
P_ = S[S["dataset"] == "prostate"]
crit["Prostate\nARI"] = per_method(P_, "ari")
crit["Prostate\ntumour F1"] = per_method(P_, "tumor_f1")
crit["ccRCC\nTLS F1"] = per_method(S[(S["dataset"] == "ccRCC") & S["tls_f1"].notna()], "tls_f1")
E = rd("results/scores_ext.csv")
if E is not None:
    E = E[(E["status"] == "ok") & E["ari"].notna() & E["method"].isin(ALLM)]
    crit["Breast\nARI"] = per_method(E, "ari")
meta = pd.read_csv(os.path.join(TAB, "sim_sections.csv")).set_index("slide")
Ssim = rd("results/scores_sim.csv")
Ssim = Ssim[(Ssim["status"] == "ok") & Ssim["ari"].notna() & Ssim["method"].isin(ALLM)]
Ps = Ssim.groupby(["slide", "method"])["ari"].mean().unstack()
crit["Simulated\nfine grain"] = Ps.loc[[s for s in Ps.index if meta.loc[s, "ell"] <= 1.0]].mean()
crit["Simulated\ncoarse grain"] = Ps.loc[[s for s in Ps.index if meta.loc[s, "ell"] >= 4.0]].mean()
MK = pd.read_csv(os.path.join(TAB, "marker_main_eta.csv")); MK = MK[MK["method"].isin(ALLM)]
w = MK.groupby(["tissue", "slide", "program", "method"])["eta2"].mean().unstack()
crit["Marker η²\nvs GMM"] = w.sub(w["gmm"], axis=0).groupby(level="tissue").mean().mean()
crit["Seed\nSD"] = pd.concat([pd.read_csv(os.path.join(TAB, f"seed_variability_{d}.csv")).set_index("method")["mean_sd"]
                              for d in ("DLPFC", "prostate")], axis=1).mean(1)
crit["Runtime\n(median)"] = S[S["dataset"].isin(["DLPFC", "prostate", "ccRCC"])].groupby("method")["sec"].median()
gpu = S.groupby("method")["gpu_peak_mb"].median().reindex(ALLM).fillna(0) > 25
LOWER = {"Seed\nSD", "Runtime\n(median)"}
C = pd.DataFrame(crit).reindex(ALLM)
Z = C.copy()
for col in C:
    v = C[col].astype(float)
    v = np.log10(v) if col.startswith("Runtime") else v
    z = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0 + 0.5
    Z[col] = 1 - z if col in LOWER else z
ACC = [c for c in C.columns if c not in LOWER]
rank = C[ACC].rank(ascending=False).mean(1)
order = rank.sort_values().index.tolist()
out = C.copy(); out.columns = [c.replace("\n", " ") for c in out.columns]
out["GPU"] = gpu; out["mean_rank_accuracy"] = rank
out.loc[order].to_csv(os.path.join(TAB, "guide_scorecard.csv"))

fig = plt.figure(figsize=(7.4, 10.6))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.0, 1.0, 1.3, 1.3, 1.15], hspace=0.75, wspace=2.4)

# a: scorecard heatmap
ax = fig.add_subplot(gs[0:2, 0:8])
cols = list(C.columns) + ["GPU"]
A = np.column_stack([Z.loc[order].values, np.full(len(order), np.nan)])
HDR = {"Simulated\nfine grain": "Fine\ngrain", "Simulated\ncoarse grain": "Coarse\ngrain", "Marker η²\nvs GMM": "Marker\nΔη²"}
heat(ax, A, SEQ[:11], 0, 1, text=False, xt=[HDR.get(c, c) for c in cols], yt=[NAME[m] for m in order])
for i, m in enumerate(order):
    for j, c in enumerate(cols):
        if c == "GPU":
            txt, z = ("yes" if gpu[m] else "–"), 0
        else:
            v = C.loc[m, c]; z = Z.loc[m, c]
            if pd.isna(v):
                continue
            if c.startswith("Runtime"):
                txt = f"{v:.0f} s" if v < 90 else f"{v / 60:.1f} min"
            elif c.startswith("Marker"):
                txt = f"{v:+.3f}"
            elif c.startswith("Seed"):
                txt = f"{v:.3f}"
            else:
                txt = f"{v:.2f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=4.3, color=SURF if z > 0.62 else INK)
ax.xaxis.tick_top(); ax.tick_params(axis="x", labelsize=4.7, pad=1); ax.tick_params(axis="y", labelsize=5.4)
from matplotlib.transforms import offset_copy
n_lab = len([c for c in C.columns if not c.startswith(("Simulated", "Marker", "Seed", "Runtime"))])
groups = [("Labelled data", n_lab), ("Simulation", 2), ("Label-free", 1), ("Practical", 3)]
tr = offset_copy(ax.get_xaxis_transform(), fig=fig, y=15, units="points")
x0 = 0
for gname, n in groups:
    ax.plot([x0 - 0.42, x0 + n - 0.58], [1, 1], transform=tr, color=INK2, lw=0.6, clip_on=False)
    ax.text((2 * x0 + n - 1) / 2, 1, gname, transform=offset_copy(ax.get_xaxis_transform(), fig=fig, y=17, units="points"),
            ha="center", va="bottom", fontsize=5.3, color=INK)
    if x0 > 0:
        ax.axvline(x0 - 0.5, color=SURF, lw=2.2)
    x0 += n
for i, m in enumerate(order):
    ax.get_yticklabels()[i].set_color(COL[m])
ax.set_title("a   Scorecard (colour: scaled within column, darker = better)", loc="left", pad=26)

# b: mean rank over the accuracy criteria
ax = fig.add_subplot(gs[0:2, 9:12]); gridx(ax)
for i, m in enumerate(order):
    ax.barh(i, rank[m], color=COL[m], height=0.62)
    ax.text(rank[m] + 0.15, i, f"{rank[m]:.1f}", va="center", fontsize=4.8, color=INK2)
ax.set_ylim(len(order) - 0.5, -0.5); ax.set_yticks([]); ax.set_xlim(0, len(order) + 0.5)
ax.set_xticks([1, 4, 7, 10, 13][:1 + len(order) // 3])
ax.set_xlabel(f"Mean rank, {len(ACC)} accuracy\ncolumns of a (1 = best)")
ax.set_title("b   Overall", loc="left", pad=26)

# ---------------- c-e: class-level rule ----------------
CF = pd.read_csv(os.path.join(TAB, "class_f1.csv")); GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
W = CF[CF["dataset"].isin(["DLPFC", "prostate"])].pivot_table(index=["slide", "dataset", "cls"], columns="method", values="f1")
W = W.reset_index().merge(GC, on=["slide", "dataset", "cls"])
W["gain"] = W[[m for m in SP9 if m in W]].mean(1) - W["gmm"]; W["log10_n"] = np.log10(W["n_spots"])
fit = smf.mixedlm("gain ~ boundary_frac + log10_n", W, groups=W["slide"]).fit(reml=True)
fe = fit.fe_params; V = fit.cov_params().loc[fe.index, fe.index].values; b = fe.values
draws = rng.multivariate_normal(b, V, 4000)
T = rd("results/tables/ext_rule_test.csv")


def pred_band(bf, n):
    X = np.column_stack([np.ones_like(bf), bf, np.full_like(bf, np.log10(n))])
    mu = X @ b; se = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
    return mu, mu - 1.96 * se, mu + 1.96 * se


ax = fig.add_subplot(gs[2, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
for ds in ["DLPFC", "prostate"]:
    g = W[W["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["gain"], s=6, color=DCOL[ds], alpha=0.55, lw=0, zorder=2)
if T is not None:
    ax.scatter(T["boundary_frac"], T["gain9"], s=8, color=DCOL["breast"], alpha=0.75, lw=0, zorder=2)
xs = np.linspace(0, 1, 60)
for n, c in [(100, SEQ[5]), (500, SEQ[8]), (2000, SEQ[12])]:
    mu, lo, hi = pred_band(xs, n)
    ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)
    ax.plot(xs, mu, color=c, lw=1.2)
    ax.text(xs[-1] + 0.02, mu[-1], f"{n:,} spots", fontsize=4.6, color=INK2, va="center")
ax.set_xlim(0, 1.27); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3, mfc=DCOL[d], mec=DCOL[d], label=DNAME[d])
                   for d in (["DLPFC", "prostate"] + (["breast"] if T is not None else []))],
          loc="upper right", fontsize=4.5, handletextpad=0.1, borderaxespad=0.2)
ax.set_xlabel("Boundary share of the class"); ax.set_ylabel("Class F1 gain, spatial − GMM")
ax.set_title("c   Gain falls with patchiness", loc="left")

ax = fig.add_subplot(gs[2, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ns = np.logspace(1.3, 3.8, 80)
be = -(draws[:, [0]] + draws[:, [2]] * np.log10(ns)[None, :]) / draws[:, [1]]
mid = -(b[0] + b[2] * np.log10(ns)) / b[1]
ax.fill_between(ns, np.clip(np.percentile(be, 2.5, 0), 0, 1), np.clip(np.percentile(be, 97.5, 0), 0, 1), color=SEQ[2], lw=0)
ax.plot(ns, np.clip(mid, 0, 1), color=SEQ[10], lw=1.3)
pts = [(W[W["dataset"] == ds], ds, "gain") for ds in ["DLPFC", "prostate"]] + ([(T, "breast", "gain9")] if T is not None else [])
for g, ds, col in pts:
    up = g[col] > 0
    ax.scatter(g.loc[up, "n_spots"], g.loc[up, "boundary_frac"], s=9, color=DCOL[ds], lw=0, zorder=3)
    ax.scatter(g.loc[~up, "n_spots"], g.loc[~up, "boundary_frac"], s=9, color=SURF, edgecolor=DCOL[ds], lw=0.6, zorder=3)
ax.set_xscale("log"); ax.set_xlim(20, 6000); ax.set_ylim(0, 1)
ax.text(0.04, 0.05, "spatial methods help", transform=ax.transAxes, fontsize=5, color=SEQ[10], fontweight="bold")
ax.text(0.96, 0.93, "GMM as good or better", transform=ax.transAxes, fontsize=5, color=INK2, ha="right", fontweight="bold")
ax.set_xlabel("Class size (spots)"); ax.set_ylabel("Boundary share of the class")
ax.text(0.96, 0.86, "●  gain > 0    ○  gain ≤ 0", transform=ax.transAxes, ha="right", fontsize=4.6, color=INK2,
        bbox=dict(fc=SURF, ec="none", pad=0.8), zorder=6)
ax.set_title("d   Break-even line (95% CI)", loc="left")
BE = pd.DataFrame({"n_spots": [50, 100, 200, 500, 1000, 2000, 5000]})
for q, nm in [(50, "break_even"), (2.5, "ci_lo"), (97.5, "ci_hi")]:
    bb = -(draws[:, [0]] + draws[:, [2]] * np.log10(BE["n_spots"].values)[None, :]) / draws[:, [1]]
    BE[nm] = np.percentile(bb, q, 0) if nm != "break_even" else -(b[0] + b[2] * np.log10(BE["n_spots"])) / b[1]
BE.to_csv(os.path.join(TAB, "guide_breakeven.csv"), index=False)

ax = fig.add_subplot(gs[2, 8:12]); gridy(ax)
W["pred"] = fe["Intercept"] + fe["boundary_frac"] * W["boundary_frac"] + fe["log10_n"] * W["log10_n"]   # fixed effects only
bars = [(ds, W[W["dataset"] == ds], "gain", "pred") for ds in ["DLPFC", "prostate"]]
if T is not None:
    bars.append(("breast", T, "gain9", "pred"))
for i, (ds, g, oc, pc) in enumerate(bars):
    agree = float((np.sign(g[oc]) == np.sign(g[pc])).mean()); r_ = spearmanr(g[pc], g[oc])[0]
    ax.bar(i, agree * 100, color=DCOL[ds], width=0.62)
    ax.text(i, agree * 100 + 2, f"{agree:.0%}\nρ {r_:.2f}\nn {len(g)}", ha="center", va="bottom", fontsize=4.6, color=INK2,
            bbox=dict(fc=SURF, ec="none", pad=0.6), zorder=4)
ax.axhline(50, color=MUTED, lw=0.7, ls="--")
ax.set_xticks(range(len(bars))); ax.set_xticklabels([{"DLPFC": "DLPFC", "prostate": "Prostate", "breast": "Breast\n(held out)"}[d]
                                                     for d, *_ in bars], fontsize=5)
ax.set_ylim(0, 125); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_ylabel("Sign of gain predicted correctly (%)")
ax.set_title("e   Rule accuracy", loc="left")

# ---------------- f: how much smoothing ----------------
LPs = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")).set_index("slide")
SG = rd("results/tables/sim_geometry.csv")
D0 = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv")); D0 = D0[D0["dataset"].isin(["DLPFC", "prostate"])]
DS = rd("results/dose/sim_dose.csv")
frames = [D0[["slide", "ring", "alpha", "seed", "ari"]].assign(bf=D0["slide"].map(LPs["boundary_frac"]))]
if DS is not None and SG is not None:
    DS = DS[DS["variant"] == "all"]
    frames.append(DS[["slide", "ring", "alpha", "seed", "ari"]].assign(bf=DS["slide"].map(SG.set_index("slide")["boundary_frac"])))
DD = pd.concat(frames).dropna(subset=["bf"])
base = DD[DD["alpha"] == 0].groupby("slide")["ari"].mean()
r1 = DD[DD["ring"] == 1].groupby(["slide", "alpha"])["ari"].mean().unstack()
r1[0.0] = base.reindex(r1.index); r1 = r1[sorted(r1.columns)]
dl = r1.sub(r1[0.0], axis=0)
bins = [(0, 0.2, "< 0.2"), (0.2, 0.4, "0.2–0.4"), (0.4, 1.01, "≥ 0.4")]
bfs = DD.groupby("slide")["bf"].first()
ax = fig.add_subplot(gs[3, 0:6]); gridy(ax); zero(ax)
bcol = {0: SEQ[12], 1: SEQ[8], 2: SEQ[4]}
best_alpha, hl = {}, []
for i, (lo, hi, nm) in enumerate(bins):
    for kind, ls in [("real", "-"), ("sim", "--")]:
        ss = [s for s in dl.index if lo <= bfs[s] < hi and s.startswith("SIM_") == (kind == "sim")]
        if len(ss) < 2:
            continue
        M_ = dl.loc[ss]; mu = M_.mean()
        ax.plot(mu.index, mu.values, color=bcol[i], ls=ls, lw=1.3 if kind == "real" else 0.9, marker="o" if kind == "real" else None, ms=1.8)
        if kind == "real":
            h = tdist.ppf(0.975, len(ss) - 1) * M_.std(ddof=1) / np.sqrt(len(ss))
            ax.fill_between(mu.index, mu - h, mu + h, color=bcol[i], alpha=0.13, lw=0)
            best_alpha[nm] = float(mu.idxmax())
            nsim = sum(lo <= bfs[s] < hi for s in dl.index if s.startswith("SIM_"))
            hl.append(Line2D([], [], color=bcol[i], lw=1.3, label=f"boundary share {nm} ({len(ss)} real, {nsim} sim.)"))
hl += [Line2D([], [], color=INK2, lw=1.2, label="real sections (± 95% CI)"), Line2D([], [], color=INK2, lw=0.9, ls="--", label="simulated")]
ax.set_xlim(0, 1); ax.set_xlabel("Neighbour weight α (1 ring)"); ax.set_ylabel("ΔARI vs α = 0")
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 0.32 * (ax.get_ylim()[1] - ax.get_ylim()[0]))   # room for the legend
ax.legend(handles=hl, loc="upper left", fontsize=4.5, ncol=2, handlelength=1.6, columnspacing=0.8, borderaxespad=0.2)
ax.set_title("f   How much to smooth, by label grain", loc="left")

# ---------------- g: cost vs accuracy ----------------
ax = fig.add_subplot(gs[3, 7:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
acc = Z[ACC].mean(1)
rt = C["Runtime\n(median)"]
for m in ALLM:
    ax.scatter(rt[m], acc[m], s=20, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3, marker="s" if gpu[m] else "o")
ax.set_xscale("log")
ax.set_xlim(rt[ALLM].min() / 1.8, rt[ALLM].max() * 3.0); ax.set_ylim(acc[ALLM].min() - 0.07, acc[ALLM].max() + 0.07)
leg = ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.4, mfc=INK2, mec=SURF, label="CPU"),
                         Line2D([], [], marker="s", ls="", ms=3.4, mfc=INK2, mec=SURF, label="GPU")], loc="upper right", fontsize=4.8)
# a label must also keep clear of the marker discs, not only of their centres
fig.canvas.draw()
disc = [ax.scatter([x], [y], s=60, alpha=0) for x, y in zip(rt[ALLM].values, acc[ALLM].values)]
place_labels(ax, fig, rt[ALLM].values, acc[ALLM].values, [NAME[m] for m in ALLM], fs=4.6, avoid=[leg] + disc)
ax.set_xlabel("Median runtime per section (s, log scale)"); ax.set_ylabel("Mean scaled accuracy (a)")
ax.set_title("g   Cost vs accuracy", loc="left")

# ---------------- h: decision guide ----------------
ax = fig.add_subplot(gs[4, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
top_coarse = [NAME[m] for m in order if m in SPM][:3]   # best spatial methods by mean rank (panel b)
b100, b500, b2000 = [float(BE.loc[BE["n_spots"] == n, "break_even"].iloc[0]) for n in (100, 500, 2000)]
a_p = best_alpha.get("≥ 0.4", 0.0)
patchy = (f"Use GMM on PCs, or smooth\nlightly (α ≈ {a_p:g}, 1 ring)" if a_p > 0 else "Use GMM on PCs;\navoid neighbour smoothing")
boxes = [
    (0.006, 0.2, 0.164, 0.66, "1  Size up the target", "Typical domain size\n(spots) and boundary\nshare, from histology\nor prior work", "#f3f2ee"),
    (0.215, 0.2, 0.2, 0.66, "2  Locate it in d",
     f"Break-even boundary share\n≈ {b100:.2f} at 100 spots,\n{b500:.2f} at 500 spots,\n{b2000:.2f} at 2,000 spots", "#f3f2ee"),
    (0.46, 0.53, 0.27, 0.44, "Below the line: compact", f"Use a spatial method,\ne.g. {', '.join(top_coarse)}", SEQ[1]),
    (0.46, 0.02, 0.27, 0.44, "Above the line: patchy or large", patchy, "#fde3cf"),
    (0.775, 0.2, 0.215, 0.66, "3  Check without labels",
     "Marker-programme η² (held-\nout genes) vs GMM: a drop\nflags over-smoothing (Fig. 6).\nReport K and seed effects", "#f3f2ee")]
for x, y, wd, ht, hd, tx, fc in boxes:
    ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.004,rounding_size=0.02", fc=fc, ec=AXIS, lw=0.7))
    ax.text(x + 0.01, y + ht - 0.06, hd, fontsize=5.3, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.01, y + ht - (0.2 if ht > 0.5 else 0.19), tx, fontsize=4.9, color=INK2, va="top", linespacing=1.3)
arr = dict(arrowstyle="-|>", color=INK2, lw=0.8, mutation_scale=7)
ax.annotate("", xy=(0.215, 0.53), xytext=(0.17, 0.53), arrowprops=arr)
ax.annotate("", xy=(0.46, 0.75), xytext=(0.415, 0.62), arrowprops=arr)
ax.annotate("", xy=(0.46, 0.24), xytext=(0.415, 0.44), arrowprops=arr)
ax.annotate("", xy=(0.775, 0.62), xytext=(0.73, 0.75), arrowprops=arr)
ax.annotate("", xy=(0.775, 0.44), xytext=(0.73, 0.24), arrowprops=arr)
ax.set_title("h   Decision guide", loc="left")

fig.legend(handles=category_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=5.6, handletextpad=0.2)
save(fig, "Fig11_guide")
print(out.loc[order].round(3).to_string())
print(BE.round(3).to_string(index=False))
print("FIG11_DONE")
