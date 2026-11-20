"""81_fig7_simulation.py - Figure 7: simulation with known domains whose spatial grain is controlled (panels a-o).
Real expression profiles, simulated domain layouts (argmax of Gaussian random fields with correlation length ell).
"""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
import geom_utils as gu

SIMCOL = {"Tumour": YELLOW, "Benign": AQUA, "Stroma": "#d6d4cb", "L1": BLUE, "L3": AQUA, "L5": MAGENTA, "L6": GREEN, "WM": VIOLET}
SRC = {"prostate": "prostate profiles", "dlpfc": "DLPFC profiles"}
rng = np.random.default_rng(5)

meta = pd.read_csv(os.path.join(TAB, "sim_sections.csv")).set_index("slide")
S = pd.read_csv(os.path.join(bu.BASE, "results/scores_sim.csv")); S = S[(S["status"] == "ok") & S["ari"].notna()]
ALLM = [m for m in dict.fromkeys(METHODS + ["bass", "precast"]) if m in S["method"].unique()]
SPM = [m for m in ALLM if m not in ("gmm", "leiden")]
P = S[S["method"].isin(ALLM)].groupby(["slide", "method"])["ari"].mean().unstack()
geo = pd.DataFrame([gu.geometry(s, "sim")[0] for s in meta.index]).set_index("slide")
G = meta.join(geo[["boundary_frac", "spot_weighted_patch"]]).join(P)
G["gain"] = G[SPM].mean(1) - G["gmm"]          # (the full table is written by 72_followup_tables.py: sim_geometry.csv)
lev = G.groupby(["source", "ell"])

# real sections on the same axes
S0 = scores(); S0 = S0[S0["dataset"].isin(["DLPFC", "prostate"]) & S0["method"].isin(METHODS)]
P0 = S0.groupby(["slide", "method"])["ari"].mean().unstack()
LPs = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")).set_index("slide")
R = LPs.loc[P0.index, ["dataset", "boundary_frac"]].copy()
R["gain"] = P0[[m for m in SP9 if m in P0]].mean(1) - P0["gmm"]
if os.path.exists(os.path.join(TAB, "ext_geometry_slide.csv")):
    X = pd.read_csv(os.path.join(TAB, "ext_geometry_slide.csv")).set_index("slide")
    X["gain"] = X[[m for m in SP9 if m in X]].mean(1) - X["gmm"]
    R = pd.concat([R, X.assign(dataset="breast")[["dataset", "boundary_frac", "gain"]]])
DCOL = {"DLPFC": MUTED, "prostate": VIOLET, "breast": MAGENTA}
DNAME = {"DLPFC": "DLPFC", "prostate": "Prostate cancer", "breast": "Breast cancer"}

fig = plt.figure(figsize=(7.4, 10.6))
gs = GridSpec(5, 12, figure=fig, height_ratios=[1.25, 1.45, 1.35, 1.35, 1.3], hspace=0.78, wspace=2.4)

# a-c: simulated truth at three grains
map_axes = []
for j, (ell, lt) in enumerate([(0.6, "a"), (1.6, "b"), (6.5, "c")]):
    sid = f"SIM_prostate_l{int(round(10 * ell)):02d}_r1"
    lab, xy = load_section(sid)
    ax = fig.add_subplot(gs[0, j * 4:(j + 1) * 4]); map_axes.append(ax)
    spatial(ax, xy.values, [SIMCOL[v] for v in lab.values], fig)
    ax.set_title(f"{lt}   ℓ = {ell:g} spacings: {G.loc[sid, 'boundary_frac']:.0%} boundary", loc="left", fontsize=6.2)
legend_below(fig, map_axes, [Line2D([], [], marker="o", ls="", ms=4.5, mfc=SIMCOL[c], mec=SURF, label=c + " profiles")
                             for c in ["Tumour", "Benign", "Stroma"]], gap=0.004)

# d: grain of the simulated labels vs ell, with the real annotations for reference
ax = fig.add_subplot(gs[1, 0:3]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for src, mk in [("prostate", "o"), ("dlpfc", "s")]:
    g = lev["boundary_frac"].agg(["mean", "min", "max"]).loc[src]
    ax.plot(g.index, g["mean"], color=INK2, lw=0.9, marker=mk, ms=2.8, mfc=SURF if src == "dlpfc" else INK2, label=SRC[src])
    ax.fill_between(g.index, g["min"], g["max"], color=GRID, lw=0)
for ds in ["DLPFC", "prostate", "breast"]:
    v = R.loc[R["dataset"] == ds, "boundary_frac"]
    if len(v):
        ax.axhline(v.median(), color=DCOL[ds], lw=0.9, ls="--")
ax.set_xscale("log"); ax.set_xticks([0.6, 1, 2.5, 6.5]); ax.set_xticklabels(["0.6", "1", "2.5", "6.5"]); ax.minorticks_off()
ax.set_xlim(0.5, 7.5); ax.set_xlabel("Correlation length ℓ (spot spacings)"); ax.set_ylabel("Boundary share of labels")
ax.legend(handles=ax.get_legend_handles_labels()[0] + [Line2D([], [], color=DCOL[d], lw=0.9, ls="--", label=f"{DNAME[d]} labels")
                                                      for d in ["DLPFC", "prostate", "breast"] if (R["dataset"] == d).any()],
          loc="lower left", fontsize=4.3, handlelength=1.4, frameon=True, framealpha=0.92, edgecolor="none", borderaxespad=0.2)
ax.set_ylim(0, 1.06)
ax.set_title("d   Grain of the truth", loc="left")

# e, f: ARI per method and grain (heatmaps)
order = G.groupby("source")[ALLM].mean().mean().sort_values(ascending=False).index.tolist()
for c0, c1, src, lt in [(4, 8, "prostate", "e"), (8, 12, "dlpfc", "f")]:
    ax = fig.add_subplot(gs[1, c0:c1])
    M = lev[ALLM].mean().loc[src][order].T
    heat(ax, M.values, SEQ[:11], 0, 0.9, fmt="{:.2f}", fs=3.9, xt=[f"{e:g}" for e in M.columns],
         yt=[NAME.get(m, m) for m in order] if c0 == 4 else [])
    ax.tick_params(axis="y", labelsize=4.8); ax.tick_params(axis="x", labelsize=5)
    ax.set_xlabel("ℓ (spot spacings)"); ax.set_title(f"{lt}   ARI, {SRC[src]}", loc="left")

# g: spatial gain vs boundary share, simulation and real sections
ax = fig.add_subplot(gs[2, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax)
for src, mk in [("prostate", "o"), ("dlpfc", "s")]:
    g = G[G["source"] == src]
    ax.scatter(g["boundary_frac"], g["gain"], s=10, marker=mk, color=SURF, edgecolor=INK2, lw=0.6, zorder=3, label=f"simulated, {SRC[src]}")
for ds in ["DLPFC", "prostate", "breast"]:
    g = R[R["dataset"] == ds]
    if len(g):
        ax.scatter(g["boundary_frac"], g["gain"], s=13, color=DCOL[ds], edgecolor=SURF, lw=0.4, zorder=4, label=DNAME[ds])
r_s = spearmanr(G["boundary_frac"], G["gain"]); r_r = spearmanr(R["boundary_frac"], R["gain"])
ax.set_xlabel("Boundary share of the reference labels"); ax.set_ylabel("ARI, spatial (mean) − GMM")
ax.set_ylim(-0.6, ax.get_ylim()[1])   # room for the legend below the points
ax.legend(loc="lower left", fontsize=4.3, handletextpad=0.1, ncol=2, columnspacing=0.6, borderaxespad=0.2)
ax.set_title(f"g   Simulated ρ = {r_s[0]:.2f}, real ρ = {r_r[0]:.2f}", loc="left")


# h: crossover grain per method (interpolated on the level means; bootstrap over replicates)
def cross(bx, by):
    o = np.argsort(bx); bx, by = np.asarray(bx)[o], np.asarray(by)[o]
    for i in range(len(bx) - 1):
        if by[i] > 0 >= by[i + 1] or by[i] <= 0 < by[i + 1]:
            return bx[i] + (bx[i + 1] - bx[i]) * by[i] / (by[i] - by[i + 1])
    return np.nan if (by > 0).any() and (by <= 0).any() else (1.0 if (by > 0).all() else 0.0)


rows = []
for src in ["prostate", "dlpfc"]:
    g = G[G["source"] == src]
    for m in SPM:
        d = (g[m] - g["gmm"]).groupby(g["ell"]).mean(); b = g.groupby("ell")["boundary_frac"].mean()
        est = cross(b.values, d.values)
        bs = []
        groups = [grp for _, grp in g.groupby("ell")]
        for _ in range(300):
            bb, dd = [], []
            for grp in groups:
                ii = rng.integers(0, len(grp), len(grp))
                bb.append(grp["boundary_frac"].values[ii].mean()); dd.append((grp[m] - grp["gmm"]).values[ii].mean())
            bs.append(cross(np.array(bb), np.array(dd)))
        rows.append(dict(source=src, method=m, crossover=est, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
CR = pd.DataFrame(rows); CR.to_csv(os.path.join(TAB, "sim_crossover.csv"), index=False)
ax = fig.add_subplot(gs[2, 5:9]); gridx(ax)
om = CR[CR["source"] == "prostate"].set_index("method")["crossover"].sort_values(ascending=False).index.tolist()
for i, m in enumerate(om):
    y = len(om) - 1 - i
    for src, off, fill in [("prostate", 0.15, True), ("dlpfc", -0.15, False)]:
        r = CR[(CR["source"] == src) & (CR["method"] == m)].iloc[0]
        ax.plot([r["lo"], r["hi"]], [y + off] * 2, color=COL.get(m, BLUE), lw=0.9)
        ax.scatter(r["crossover"], y + off, s=12, color=COL.get(m, BLUE) if fill else SURF, edgecolor=COL.get(m, BLUE), lw=0.8, zorder=3)
ax.set_yticks(range(len(om))); ax.set_yticklabels([NAME.get(m, {"bass": "BASS", "precast": "PRECAST"}.get(m, m)) for m in om][::-1], fontsize=4.8)
ax.set_ylim(-0.6, len(om) - 0.4); ax.set_xlim(0, 1)
for ds in ["DLPFC", "prostate"]:
    ax.axvline(R.loc[R["dataset"] == ds, "boundary_frac"].median(), color=DCOL[ds], lw=0.8, ls="--", zorder=1)
ax.set_xlabel("Boundary share where the gain over GMM ends")
ax.text(0.02, 0.98, "●  prostate profiles\n○  DLPFC profiles", transform=ax.transAxes, ha="left", va="top", fontsize=4.6,
        color=INK2, bbox=dict(fc=SURF, ec="none", pad=0.8), zorder=6)
ax.set_title("h   Break-even grain", loc="left")

# i: ranks at the finest vs coarsest grain
ax = fig.add_subplot(gs[2, 9:12])
fine = G[G["ell"] == 0.6][ALLM].mean().rank(ascending=False); coarse = G[G["ell"] == 6.5][ALLM].mean().rank(ascending=False)
for m in ALLM:
    c = COL.get(m, BLUE)
    ax.plot([0, 1], [fine[m], coarse[m]], color=c, lw=1.0 if m in ("gmm", "smooth") else 0.7, alpha=1 if m in ("gmm", "smooth") else 0.7)
    ax.text(-0.04, fine[m], NAME.get(m, m.upper()), ha="right", va="center", fontsize=4.4, color=INK2)
    ax.text(1.04, coarse[m], NAME.get(m, m.upper()), ha="left", va="center", fontsize=4.4, color=INK2)
ax.set_ylim(len(ALLM) + 0.6, 0.4); ax.set_xlim(-0.95, 1.95); ax.set_xticks([0, 1]); ax.set_xticklabels(["ℓ 0.6", "ℓ 6.5"])
ax.set_yticks([]); [ax.spines[s_].set_visible(False) for s_ in ["left", "bottom"]]
ax.set_title("i   Rank flips", loc="center")

# j-l: PCA-smooth dose on the simulated sections
DS = pd.read_csv(os.path.join(bu.BASE, "results/dose/sim_dose.csv")) if os.path.exists(os.path.join(bu.BASE, "results/dose/sim_dose.csv")) else None
if DS is not None:
    DS = DS.merge(meta[["source", "ell"]], left_on="slide", right_index=True)
    base = DS[DS["alpha"] == 0].groupby("slide")["ari"].mean()
    ax = fig.add_subplot(gs[3, 0:4]); gridy(ax); zero(ax)
    ells = sorted(DS["ell"].unique()); cseq = [SEQ[i] for i in np.linspace(3, 12, len(ells)).astype(int)]
    for e, c in zip(ells, cseq):
        d = DS[(DS["source"] == "prostate") & (DS["ell"] == e) & (DS["ring"].isin([0, 1]))]
        w_ = d.groupby(["slide", "alpha"])["ari"].mean().unstack()
        cur = w_.sub(base.reindex(w_.index), axis=0).mean()
        ax.plot(cur.index, cur.values, color=c, lw=1.1, marker="o", ms=1.6)
        ax.text(1.03, cur.values[-1], f"ℓ {e:g}", fontsize=4.5, color=INK2, va="center")
    ax.set_xlim(-0.03, 1.25); ax.set_xticks([0, 0.5, 1]); ax.set_xlabel("Neighbour weight α (1 ring)")
    ax.set_ylabel("ΔARI vs α = 0"); ax.set_title("j   Smoothing dose, prostate profiles", loc="left")
    # k: best alpha vs boundary share, simulation and real
    ax = fig.add_subplot(gs[3, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    b1 = DS[DS["ring"].isin([0, 1])].groupby(["slide", "alpha"])["ari"].mean().unstack().idxmax(1)
    for src, mk in [("prostate", "o"), ("dlpfc", "s")]:
        ss = meta.index[meta["source"] == src]
        ax.scatter(G.loc[ss, "boundary_frac"], b1.loc[ss] + rng.uniform(-0.015, 0.015, len(ss)), s=9, marker=mk, color=SURF,
                   edgecolor=INK2, lw=0.6, zorder=3)
    D0 = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
    b0 = D0[D0["ring"].isin([0, 1])].groupby(["slide", "alpha"])["ari"].mean().unstack().idxmax(1)
    for ds in ["DLPFC", "prostate"]:
        ss = [s for s in b0.index if s in LPs.index and LPs.loc[s, "dataset"] == ds]
        ax.scatter(LPs.loc[ss, "boundary_frac"], b0.loc[ss] + rng.uniform(-0.015, 0.015, len(ss)), s=12, color=DCOL[ds],
                   edgecolor=SURF, lw=0.4, zorder=4)
    allb = pd.concat([G["boundary_frac"], LPs.loc[[s for s in b0.index if s in LPs.index], "boundary_frac"]])
    allv = pd.concat([b1.loc[G.index], b0.loc[[s for s in b0.index if s in LPs.index]]])
    r_ = spearmanr(allb, allv)
    ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("Best α (1 ring)")
    ax.set_title(f"k   Best α vs grain (ρ = {r_[0]:.2f})", loc="left")
    # l: neighbourhood size at full smoothing
    ax = fig.add_subplot(gs[3, 8:12]); gridy(ax); zero(ax)
    for ring, ls in [(1, "-"), (2, "--"), (3, ":")]:
        d = DS[(DS["alpha"] == 1.0) & (DS["ring"] == ring)].groupby("slide")["ari"].mean() - base
        t = pd.DataFrame({"d": d, "b": G.loc[d.index, "boundary_frac"], "ell": G.loc[d.index, "ell"]}).groupby("ell").mean()
        ax.plot(t["b"], t["d"], color=INK, ls=ls, lw=1.0, marker="o", ms=1.8, label=f"{ring} ring{'s' if ring > 1 else ''}")
    ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("ΔARI at α = 1")
    ax.legend(loc="lower left", fontsize=4.8); ax.set_title("l   Wider neighbourhoods", loc="left")

# m-o: domain maps: the same deep spatial method at fine and coarse grain, against GMM (prostate profiles)
from scipy.optimize import linear_sum_assignment as lsa
cls = ["Tumour", "Benign", "Stroma"]
for j, (sid, m, lt) in enumerate([("SIM_prostate_l10_r1", "gmm", "m"), ("SIM_prostate_l10_r1", "stagate", "n"),
                                  ("SIM_prostate_l65_r1", "stagate", "o")]):
    lab, xy = load_section(sid)
    ax = fig.add_subplot(gs[4, j * 4:(j + 1) * 4])
    pr = run_labels(sid, m, 3, 0, os.path.join(bu.BASE, "results/runs_sim")).reindex(xy.index)
    ucl = np.unique(pr.values)
    C = np.array([[np.sum((pr.values == u) & (lab.values == c)) for c in cls] for u in ucl])
    r, c = lsa(-C); mpc = {ucl[i]: SIMCOL[cls[k]] for i, k in zip(r, c)}
    spatial(ax, xy.values, [mpc.get(v, MUTED) for v in pr.values], fig)
    ax.set_title(f"{lt}   {NAME[m]}, ℓ = {meta.loc[sid, 'ell']:g}: ARI {G.loc[sid, m]:.2f}", loc="left", fontsize=6.2, fontweight="normal")
fig.legend(handles=category_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=3, fontsize=5.6, handletextpad=0.2)
save(fig, "Fig7_simulation")
print("FIG7SIM_DONE")
