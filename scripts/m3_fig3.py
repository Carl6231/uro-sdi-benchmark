"""m3_fig3.py - main Figure 3: the grain of the reference labels explains when spatial methods gain (panels a-p)."""
from mstyle import *
from collections import Counter
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment as lsa
import statsmodels.formula.api as smf

GS = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")); GC = pd.read_csv(os.path.join(TAB, "label_patches_class.csv"))
CF = pd.read_csv(os.path.join(TAB, "class_f1.csv")); CR = pd.read_csv(os.path.join(TAB, "class_gain_vs_patchiness.csv"))
MS = pd.read_csv(os.path.join(TAB, "majority_filter.csv")); ML = pd.read_csv(os.path.join(TAB, "majority_filter_labels.csv"))
SG = pd.read_csv(os.path.join(TAB, "sim_geometry.csv")).set_index("slide")
SS = pd.read_csv(os.path.join(bu.BASE, "results/scores_sim.csv")); SS = SS[(SS["status"] == "ok") & SS["ari"].notna()]
rng = np.random.default_rng(0)
DCOL = {"DLPFC": MUTED, "prostate": VIOLET, "ccRCC": INK, "breast": MAGENTA}
DNAME = {"DLPFC": "DLPFC", "prostate": "Prostate cancer", "ccRCC": "ccRCC (TLS)", "breast": "Breast cancer"}
SIMCOL = {"Tumour": YELLOW, "Benign": AQUA, "Stroma": "#d6d4cb"}
BDOT = Line2D([], [], marker="o", ls="", ms=2.2, mfc=INK, mec=INK, label="boundary spot (touches another label)")


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


def label_map(ax, xy, v, nb):
    bnd = boundary(v, nb)
    spatial(ax, xy, label_colors(v), fig)
    ax.scatter(xy[bnd, 0], xy[bnd, 1], s=0.45, color=INK, lw=0, zorder=4)
    return bnd


def note(ax, txt):
    ax.text(0.5, -0.02, txt, transform=ax.transAxes, ha="center", va="top", fontsize=FMIN, color=INK2)


fig = plt.figure(figsize=(FW, 9.45))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.45, 1.45, 1.3, 1.35, 1.15], hspace=0.72, wspace=2.2)

# a-c: reference labels with their boundary spots
maps0 = []
for i, (k, ttl) in enumerate([("151673", "DLPFC 151673"), ("H1_2", "Prostate H1_2"), ("H2_1", "Prostate H2_1")]):
    sid = next(s for s in GS["slide"] if s.endswith(k))
    ax = fig.add_subplot(gs[0, i * 4:(i + 1) * 4]); maps0.append(ax)
    lab, xy = load_section(sid); nb, _ = bu.lattice_neighbors(xy.values)
    label_map(ax, xy.values, lab.values, nb)
    vc = lab.value_counts(); direct_labels(ax, xy.values, lab.values, [c for c in vc.index if vc[c] >= 300], fs=FMIN)
    r = GS.set_index("slide").loc[sid]
    ptitle(ax, "abc"[i], ttl)
    note(ax, f"{r['boundary_frac'] * 100:.0f}% boundary spots · {int(r['n_patches'])} patches")

# d: boundary share per section; e: class-level gain (mixed model); f: per-method correlation
ax = fig.add_subplot(gs[1, 0:4]); gridx(ax)
X = pd.read_csv(os.path.join(TAB, "ext_geometry_slide.csv"))
rows = [("ccRCC", GS.loc[GS["dataset"] == "ccRCC", "boundary_frac"].values), ("breast", X["boundary_frac"].values),
        ("prostate", GS.loc[GS["dataset"] == "prostate", "boundary_frac"].values), ("DLPFC", GS.loc[GS["dataset"] == "DLPFC", "boundary_frac"].values)]
for yy, (ds, v) in enumerate(rows):
    ax.scatter(v, yy + rng.uniform(-0.16, 0.16, len(v)), s=10, color=DCOL[ds], alpha=0.8, lw=0, zorder=3)
    ax.plot([np.median(v)] * 2, [yy - 0.32, yy + 0.32], color=INK, lw=1.1, zorder=4)
    ax.text(np.median(v), yy + 0.36, f"{np.median(v):.2f}", ha="center", va="bottom", fontsize=FMIN, color=INK2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([DNAME[d].replace(" cancer", "") for d, _ in rows])
ax.set_ylim(-0.5, len(rows) - 0.25); ax.set_xlabel("Share of boundary spots"); ptitle(ax, "d", "Label grain per section")

ax = fig.add_subplot(gs[1, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
W = CF.pivot_table(index=["slide", "dataset", "cls"], columns="method", values="f1").reset_index().merge(GC, on=["slide", "dataset", "cls"])
W["gain9"] = W[[m for m in SP9 if m in W]].mean(1) - W["gmm"]; W["log10_n"] = np.log10(W["n_spots"])
for ds in ["DLPFC", "prostate"]:
    g = W[W["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["gain9"], s=np.clip(g["n_spots"] / 60, 5, 40), color=DCOL[ds], alpha=0.7, edgecolor=SURF, lw=0.3, zorder=3)
fit = smf.mixedlm("gain9 ~ boundary_frac + log10_n", W, groups=W["slide"]).fit(reml=True)
xs = np.linspace(0, W["boundary_frac"].max(), 40)
for n, ls in [(100, ":"), (500, "-"), (2000, "--")]:
    ax.plot(xs, fit.params["Intercept"] + fit.params["boundary_frac"] * xs + fit.params["log10_n"] * np.log10(n), color=INK, lw=1.0, ls=ls,
            label=f"{n:,} spots")
pv = fit.pvalues["boundary_frac"]
ax.legend(loc="upper right", fontsize=FMIN, handlelength=1.6, borderaxespad=0.1, title="model, class of", title_fontsize=FMIN,
          frameon=True, framealpha=0.9, edgecolor="none")
ax.set_xlabel("Boundary share of the class"); ax.set_ylabel("ΔF1, spatial − GMM")
ptitle(ax, "e", f"{len(W)} classes (slope P = {pv:.3f})")

ax = fig.add_subplot(gs[1, 8:11]); gridx(ax); zero(ax, "x")
cr = CR[CR["scope"] == "all"].set_index("method")
o = [m for m in cr.sort_values("rho_vs_boundary").index if m in SP9] + ["spatial9"]
for i, m in enumerate(o):
    r = cr.loc[m]; y = len(o) - 1 - i; c = COL.get(m, INK)
    ax.scatter(r["rho_vs_boundary"], y, s=18, color=c if r["p"] < 0.05 else SURF, edgecolor=c, lw=0.9, zorder=3)
ax.set_yticks(range(len(o))); ax.set_yticklabels([NAME.get(m, f"mean of {NSP}") for m in o][::-1], fontsize=FMIN)
ax.yaxis.tick_right(); ax.set_ylim(-0.6, len(o) - 0.4)
ax.set_xlabel("ρ, class gain vs grain"); ptitle(ax, "f", "Per method")
ax.text(0.55, 0.97, "● P < 0.05\n○ P ≥ 0.05", transform=ax.transAxes, fontsize=FMIN, color=INK2, va="top")

# g-i: majority vote applied to the labels of one prostate section
ch = ML[(ML["dataset"] == "prostate") & (ML["level"] == 1)].set_index("slide")["changed"]
sid = ch.idxmax()
l0, xy = load_section(sid); xyv = xy.values
nb1, _ = bu.lattice_neighbors(xyv, 1.1); nb2, _ = bu.lattice_neighbors(xyv, 2.1)
maps3 = []
for i, (k, v) in enumerate([("original labels", l0.values), ("1-ring majority vote", majority(l0.values, nb1)),
                            ("2-ring majority vote", majority(l0.values, nb2))]):
    ax = fig.add_subplot(gs[2, i * 4:(i + 1) * 4]); maps3.append(ax)
    bnd = label_map(ax, xyv, v, nb1)
    s = pd.notna(v)
    ptitle(ax, "ghi"[i], f"{short(sid)}, {k}", fontsize=FS)
    note(ax, f"{bnd[s].mean() * 100:.0f}% boundary" + (f" · {np.mean(v[s] != l0.values[s]) * 100:.0f}% relabelled" if i else ""))
present = [c for c in l0.value_counts().index if l0.value_counts()[c] >= 5]
maps_legend(fig, maps3, cls_handles(present) + [BDOT], gap=0.018)

# j: re-scoring; k: methods above GMM; l: gain vs boundary share, simulated and real sections
ax = fig.add_subplot(gs[3, 0:4]); gridy(ax)
ms = MS[MS["dataset"] == "prostate"]
for m in [x for x in METHODS if x != "gmm"] + ["gmm"]:
    g = ms[ms["method"] == m].sort_values("level")
    lw = 2.0 if m == "gmm" else 1.4 if m in ("smooth", "graphst") else 0.8
    ax.plot(g["level"], g["mean_ari"], color=COL[m], lw=lw, marker="o", ms=2.5, alpha=1 if lw > 1 else 0.55, zorder=4 if m == "gmm" else 2)
for m in ["gmm", "graphst"]:
    g = ms[ms["method"] == m].sort_values("level"); ax.text(2.08, g["mean_ari"].iloc[-1], NAME[m], fontsize=FMIN, va="center", color=INK2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["original", "1 ring", "2 rings"]); ax.set_xlim(-0.15, 2.7)
ax.set_xlabel("Prostate labels after majority vote"); ax.set_ylabel("Mean ARI"); ptitle(ax, "j", "Re-scored, same clusters")

ax = fig.add_subplot(gs[3, 4:8]); gridy(ax)
for j, ds in enumerate(["DLPFC", "prostate"]):
    m_ = MS[(MS["dataset"] == ds) & MS["method"].isin(SP9)]
    cnt = m_.groupby("level")["diff_vs_gmm"].apply(lambda x: int((x > 0).sum()))
    bfr = ML[ML["dataset"] == ds].groupby("level")["boundary_frac"].mean()
    for lev in [0, 1, 2]:
        x = lev + (j - 0.5) * 0.38
        ax.bar(x, cnt[lev], width=0.36, color=DCOL[ds])
        ax.text(x, cnt[lev] + 0.25, f"{bfr[lev]:.2f}", ha="center", fontsize=FMIN, color=INK2)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["original", "1 ring", "2 rings"]); ax.set_ylim(0, NSP + 2)
ax.text(0.98, 0.97, "on bars: boundary share", transform=ax.transAxes, ha="right", va="top", fontsize=FMIN, color=INK2)
ax.set_ylabel(f"Methods above GMM (of {NSP})"); ax.set_xlabel("Labels after majority vote")
ptitle(ax, "k", "Smoother labels, spatial wins")

ax = fig.add_subplot(gs[3, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
S0 = scores(); S0 = S0[S0["dataset"].isin(["DLPFC", "prostate"]) & S0["method"].isin(METHODS)]
P0 = S0.groupby(["slide", "method"])["ari"].mean().unstack()
LPs = GS.set_index("slide")
R = LPs.loc[P0.index, ["dataset", "boundary_frac"]].copy(); R["gain"] = P0[[m for m in SP9 if m in P0]].mean(1) - P0["gmm"]
Xs = X.set_index("slide"); Xs["gain"] = Xs[[m for m in SP9 if m in Xs]].mean(1) - Xs["gmm"]
R = pd.concat([R, Xs.assign(dataset="breast")[["dataset", "boundary_frac", "gain"]]])
for src, mk in [("prostate", "o"), ("dlpfc", "s")]:
    g = SG[SG["source"] == src]
    ax.scatter(g["boundary_frac"], g["gain9"], s=11, marker=mk, color=SURF, edgecolor=INK2, lw=0.6, zorder=3)
for ds in ["DLPFC", "prostate", "breast"]:
    g = R[R["dataset"] == ds]
    ax.scatter(g["boundary_frac"], g["gain"], s=14, color=DCOL[ds], edgecolor=SURF, lw=0.4, zorder=4)
r_s = spearmanr(SG["boundary_frac"], SG["gain9"]); r_r = spearmanr(R["boundary_frac"], R["gain"])
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("ΔARI, spatial − GMM")
ax.set_ylim(-0.72, 0.5)
ax.text(0.03, 0.03, f"○ □ simulated, ρ = {r_s[0]:.2f}\n●  real, ρ = {r_r[0]:.2f}", transform=ax.transAxes, ha="left",
        va="bottom", fontsize=FMIN, color=INK2)
ptitle(ax, "l", "Simulated and real sections")

# m-p: simulation with real prostate profiles: truth and STAGATE at fine and coarse grain
cls = ["Tumour", "Benign", "Stroma"]
smaps = []
for j, (sid, m, lt) in enumerate([("SIM_prostate_l10_r1", None, "m"), ("SIM_prostate_l10_r1", "stagate", "n"),
                                  ("SIM_prostate_l65_r1", None, "o"), ("SIM_prostate_l65_r1", "stagate", "p")]):
    lab, xy = load_section(sid)
    ax = fig.add_subplot(gs[4, j * 3:(j + 1) * 3]); smaps.append(ax)
    ell = SG.loc[sid, "ell"]
    if m is None:
        spatial(ax, xy.values, [SIMCOL[v] for v in lab.values], fig)
        ptitle(ax, lt, f"Truth, ℓ = {ell:g} ({SG.loc[sid, 'boundary_frac']:.0%})", fontsize=FS)
        continue
    pr = run_labels(sid, m, 3, 0, os.path.join(bu.BASE, "results/runs_sim")).reindex(xy.index)
    ucl = np.unique(pr.values)
    C = np.array([[np.sum((pr.values == u) & (lab.values == c)) for c in cls] for u in ucl])
    r, c = lsa(-C); mpc = {ucl[i]: SIMCOL[cls[k]] for i, k in zip(r, c)}
    spatial(ax, xy.values, [mpc.get(v, MUTED) for v in pr.values], fig)
    a_ = SS[(SS["slide"] == sid) & (SS["method"] == m)]["ari"].mean()
    ptitle(ax, lt, f"{NAME[m]}  ·  ARI {a_:.2f}", fontsize=FS)
maps_legend(fig, smaps, [Line2D([], [], marker="o", ls="", ms=5, mfc=SIMCOL[c], mec=SURF, label=f"{c} profiles") for c in cls],
            gap=0.004)
top_legend(fig, [Line2D([], [], marker="o", ls="", ms=5, mfc=DCOL[d], mec=SURF, label=DNAME[d]) for d in ["DLPFC", "prostate", "breast", "ccRCC"]],
           y=0.925)
msave(fig, "Figure3")
print("M3_DONE")
