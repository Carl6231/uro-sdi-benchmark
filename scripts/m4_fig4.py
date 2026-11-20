"""m4_fig4.py - main Figure 4: the smoothing dose, and the neighbour-pair errors it corrects and introduces (panels a-m).
Pair tables: results/boundary (93_boundary_errors.py, 94/95 summaries). The maps in j-m recompute PCA-smooth for seed 0."""
from mstyle import *
import importlib
from scipy import sparse
from scipy.stats import spearmanr
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from sklearn.metrics import adjusted_rand_score
import geom_utils as gu

D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
DE = pd.read_csv(os.path.join(bu.BASE, "results/dose/ext_dose.csv"))
SD = scores("results/scores_dose.csv")
GS = pd.read_csv(os.path.join(TAB, "label_patches_slide.csv")).set_index("slide")
XG = pd.read_csv(os.path.join(TAB, "ext_geometry_slide.csv")).set_index("slide")
BD = os.path.join(bu.BASE, "results/boundary")
PC = pd.read_csv(os.path.join(BD, "paired_cohorts.csv")); PS = pd.read_csv(os.path.join(BD, "paired_sections.csv"))
DD = pd.read_csv(os.path.join(BD, "dose_donors.csv")); PL = pd.read_csv(os.path.join(BD, "prostate_label_pairs.csv"))
AD = pd.read_csv(os.path.join(BD, "archive_donors.csv"))
DCOL = {"DLPFC": MUTED, "prostate": VIOLET, "breast": MAGENTA}
DNAME = {"DLPFC": "DLPFC", "prostate": "Prostate", "breast": "Breast"}
GOOD, BAD = BLUE, RED                              # corrected (wrong to right), introduced (right to wrong)
rng = np.random.default_rng(0)


def ring1(ds):
    d = D[(D["dataset"] == ds) & D["ari"].notna()]
    return pd.concat([d[d["alpha"] == 0], d[d["ring"] == 1]]).groupby(["slide", "alpha"])["ari"].mean().unstack()


fig = plt.figure(figsize=(FW, 8.6))
gs = GridSpec(4, 12, figure=fig, height_ratios=[1.35, 1.3, 1.35, 1.45], hspace=0.62, wspace=2.2)

# a: PCA-smooth dose, every section and the mean
ax = fig.add_subplot(gs[0, 0:4]); gridy(ax); zero(ax)
for ds in ["DLPFC", "prostate"]:
    w = ring1(ds); dl = w.sub(w[0.0], axis=0)
    for s in dl.index:
        ax.plot(dl.columns, dl.loc[s].values, color=DCOL[ds], lw=0.5, alpha=0.45, zorder=2)
    ax.plot(dl.columns, dl.mean().values, color=DCOL[ds], lw=2.0, marker="o", ms=2.8, mec=SURF, mew=0.4, zorder=4,
            label=f"{DNAME[ds]} ({len(dl)} sections)")
ax.set_xlabel("Neighbour weight α (1 ring)"); ax.set_ylabel("ΔARI vs α = 0")
ax.legend(loc="lower left", fontsize=FMIN, borderaxespad=0.2)
ptitle(ax, "a", "PCA-smooth dose")

# b: best alpha vs label grain, three data sets
ax = fig.add_subplot(gs[0, 4:8]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
pts = []
for ds in ["DLPFC", "prostate"]:
    best = ring1(ds).idxmax(1); bf = GS.loc[best.index, "boundary_frac"]
    ax.scatter(bf, best.values + rng.uniform(-0.02, 0.02, len(best)), s=16, color=DCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
    pts.append(pd.DataFrame({"bf": bf.values, "best": best.values}))
de = DE[(DE["variant"] == "all") & DE["ari"].notna() & DE["slide"].isin(XG.index)]
best = de.groupby(["slide", "alpha"])["ari"].mean().unstack().idxmax(1); bf = XG.loc[best.index, "boundary_frac"]
ax.scatter(bf, best.values + rng.uniform(-0.02, 0.02, len(best)), s=16, color=DCOL["breast"], edgecolor=SURF, lw=0.5, zorder=3)
pts.append(pd.DataFrame({"bf": bf.values, "best": best.values}))
pp = pd.concat(pts); r_ = spearmanr(pp["bf"], pp["best"])
ax.set_xlabel("Boundary share of the labels"); ax.set_ylabel("Best α")
ptitle(ax, "b", f"Best smoothing (ρ = {r_[0]:.2f})")

# c: the same pattern in BANKSY
ax = fig.add_subplot(gs[0, 8:12]); gridy(ax); zero(ax)
b = SD[SD["method"].str.startswith("banksy-lam")].copy(); b["lev"] = b["method"].str[10:].astype(float)
for ds in ["DLPFC", "prostate"]:
    w = b[b["dataset"] == ds].groupby(["slide", "lev"])["ari"].mean().unstack(); dl = w.sub(w[0.0], axis=0)
    for s in dl.index:
        ax.plot(dl.columns, dl.loc[s].values, color=DCOL[ds], lw=0.5, alpha=0.45)
    ax.plot(dl.columns, dl.mean().values, color=DCOL[ds], lw=2.0, marker="o", ms=2.8, mec=SURF, mew=0.4, zorder=4)
ax.axvline(0.2, color=INK2, lw=0.6, ls=":"); ax.text(0.22, 0.97, "default", transform=ax.get_xaxis_transform(), fontsize=FMIN,
                                                     color=INK2, va="top")
ax.set_xlabel("BANKSY spatial weight λ"); ax.set_ylabel("ΔARI vs λ = 0")
ptitle(ax, "c", "BANKSY, same pattern")


# d, e: corrected vs introduced pair errors (equal donor weights; breast: equal section weights)
def pair_bars(ax, etype, lt, ttl):
    gridx(ax)
    for i, ds in enumerate(["DLPFC", "prostate", "breast"]):
        y = 2 - i
        for al, off, alpha_ in [(0.3, 0.17, 1.0), (0.7, -0.17, 0.45)]:
            r = PC[(PC["dataset"] == ds) & (PC["alpha"] == al) & (PC["edge_type"] == etype)].iloc[0]
            ax.barh(y + off + 0.075, r["corrected_rate"] * 100, height=0.15, color=GOOD, alpha=alpha_)
            ax.barh(y + off - 0.075, r["introduced_rate"] * 100, height=0.15, color=BAD, alpha=alpha_)
            if al == 0.3:
                for v, dy in [(r["corrected_rate"], 0.075), (r["introduced_rate"], -0.075)]:
                    ax.text(v * 100 + 0.5, y + off + dy, f"{v * 100:.1f}", va="center", fontsize=FMIN - 0.4, color=INK2)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([DNAME[d] for d in ["DLPFC", "prostate", "breast"]])
    ax.set_xlim(0, 44); ax.set_xticks([0, 10, 20, 30]); ax.set_xlabel("Share of pairs (%)"); ptitle(ax, lt, ttl)


ax = fig.add_subplot(gs[1, 0:4]); pair_bars(ax, "within", "d", "Same-label neighbours")
ax.legend(handles=[Patch(color=GOOD, label="corrected"), Patch(color=BAD, label="newly wrong"),
                   Patch(color=INK2, alpha=1, label="α = 0.3"), Patch(color=INK2, alpha=0.45, label="α = 0.7")],
          loc="lower right", fontsize=FMIN, ncol=1, handlelength=1.0, borderaxespad=0.1)
ax = fig.add_subplot(gs[1, 4:8]); pair_bars(ax, "boundary", "e", "Across a boundary")

# f: per section, change in split and merge rates at alpha = 0.3
ax = fig.add_subplot(gs[1, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True); zero(ax); zero(ax, "x")
p3 = PS[PS["alpha"] == 0.3].pivot_table(index=["dataset", "slide"], columns="edge_type", values="net_error_change").reset_index()
for ds in ["breast", "DLPFC", "prostate"]:
    g = p3[p3["dataset"] == ds]
    ax.scatter(g["within"] * 100, g["boundary"] * 100, s=18, color=DCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
ax.set_xlabel("Δ split rate, same label (pp)"); ax.set_ylabel("Δ merge rate, boundary (pp)")
n_both = int(((p3["within"] < 0) & (p3["boundary"] > 0) & p3["dataset"].isin(["DLPFC", "prostate"])).sum())
ax.set_ylim(-26, 33)
ax.text(0.03, 0.03, f"{n_both}/19 DLPFC and prostate sections:\nfewer splits, more merges", transform=ax.transAxes, va="bottom",
        fontsize=FMIN, color=INK2)
ptitle(ax, "f", "Every section, α = 0.3")

# g: whole-section ARI against the boundary merge rate along the dose, per donor
ax = fig.add_subplot(gs[2, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for (ds, dn), g in DD.groupby(["dataset", "donor"]):
    g = g.sort_values("alpha")
    ax.plot(g["merge_rate"] * 100, g["ari"], color=DCOL[ds], lw=1.3, zorder=2)
    for _, r in g.iterrows():
        ax.scatter(r["merge_rate"] * 100, r["ari"], s=18 + 30 * r["alpha"], color=DCOL[ds], edgecolor=SURF, lw=0.5, zorder=3)
    if ds == "prostate":
        for _, r in g.iterrows():
            ax.text(r["merge_rate"] * 100, r["ari"] + 0.022, f"α {r['alpha']:g}", fontsize=FMIN, ha="center", va="bottom", color=VIOLET)
ax.set_xlabel("Boundary pairs merged (%)"); ax.set_ylabel("ARI (whole section)")
ax.text(0.03, 0.97, "DLPFC donors: ARI ↑\nprostate: ARI ↓", transform=ax.transAxes, va="top", fontsize=FMIN, color=INK2)
ptitle(ax, "g", "Score up, boundaries lost")

# h: the most frequent prostate boundaries
ax = fig.add_subplot(gs[2, 4:8]); gridy(ax)
pairs = [("Benign", "Stroma"), ("GG2", "Stroma"), ("Benign*", "Stroma"), ("Benign", "GG2"), ("GG4 Cribriform", "Stroma")]
PN = {("Benign", "Stroma"): "benign–stroma", ("GG2", "Stroma"): "GG2–stroma", ("Benign*", "Stroma"): "benign*–stroma",
      ("Benign", "GG2"): "benign–GG2", ("GG4 Cribriform", "Stroma"): "GG4 crib.–stroma"}
for a_, b_ in pairs:
    r3 = PL[(PL["edge_type"] == "boundary") & (PL["class_a"] == a_) & (PL["class_b"] == b_) & (PL["alpha"] == 0.3)].iloc[0]
    r7 = PL[(PL["edge_type"] == "boundary") & (PL["class_a"] == a_) & (PL["class_b"] == b_) & (PL["alpha"] == 0.7)].iloc[0]
    y = np.array([r3["error_rate_before"], r3["error_rate_after"], r7["error_rate_after"]]) * 100
    main_ = a_ == "Benign" and b_ == "Stroma"
    ax.plot([0, 0.3, 0.7], y, color=BAD if main_ else INK2, lw=2.0 if main_ else 0.9, marker="o", ms=3 if main_ else 2, zorder=3 if main_ else 2)
    dy = {("Benign", "Stroma"): 1.5, ("GG2", "Stroma"): -0.6, ("Benign*", "Stroma"): -2.4, ("Benign", "GG2"): 0.9,
          ("GG4 Cribriform", "Stroma"): -0.9}.get((a_, b_), 0)
    ax.text(0.73, y[-1] + dy, PN[(a_, b_)], fontsize=FMIN, va="center", color=INK if main_ else INK2)
ax.set_xlim(-0.05, 1.28); ax.set_xticks([0, 0.3, 0.7]); ax.set_ylim(48, 100)
ax.set_xlabel("Neighbour weight α"); ax.set_ylabel("Pairs in one cluster (%)")
ptitle(ax, "h", "Prostate boundaries (pairs)")

# i: the same trade-off across the 13 methods (prostate, archived benchmark runs)
ax = fig.add_subplot(gs[2, 8:12]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
pr = AD[AD["dataset"] == "prostate"].set_index("method")
for m, r in pr.iterrows():
    ax.scatter(r["split_rate"] * 100, r["merge_rate"] * 100, s=24, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3)
ax.set_xlim(0.5, pr["split_rate"].max() * 100 + 3.5)
ax.set_ylim(pr["merge_rate"].min() * 100 - 3, pr["merge_rate"].max() * 100 + 3)
OFF = {"spaceflow": (4, 0, "left"), "sedr": (-4, 0, "right"), "bass": (4, 0, "left"), "smooth": (-4, 0, "right"),
       "graphst": (-4, -3, "right"), "leiden": (-4, 0, "right"), "spagcn": (0, -6, "center"), "drsc": (4, -2, "left"),
       "banksy": (0, 5, "center"), "gmm": (4, 0, "left")}      # STAGATE, BayesSpace and PRECAST sit in the spatial cloud
for m, (dx, dy, ha) in OFF.items():
    ax.annotate(NAME[m], (pr.loc[m, "split_rate"] * 100, pr.loc[m, "merge_rate"] * 100), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="top" if dy < 0 else ("bottom" if dy > 0 else "center"), fontsize=FMIN, color=COL[m] if m in ("smooth",) else INK)
ax.set_xlabel("Same-label pairs split (%)"); ax.set_ylabel("Boundary pairs merged (%)")
ptitle(ax, "i", "13 methods, prostate")

# j-m: maps of individual pairs that changed between alpha = 0 and 0.3 (seed 0)
dose = importlib.import_module("63_dose_generic")
RD = pd.read_csv(os.path.join(bu.BASE, "results/dose/real_dose.csv"))


def recompute(sid):
    a = bu.load_slide(sid); labels = bu.scored_labels(a)
    ref = labels.fillna("").to_numpy(); valid = labels.notna().to_numpy()
    xy = np.asarray(a.obsm["spatial"], float)
    _, row, col = gu.ring_edges(xy, 1.1)
    g = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(xy), len(xy)))
    z = dose.pcs(a); deg = np.asarray(g.sum(1)).ravel()
    nz = np.asarray(sparse.diags(1. / np.maximum(deg, 1)) @ g @ z); nz[deg == 0] = z[deg == 0]
    k = int(RD.loc[RD["slide"] == sid, "k"].iloc[0])
    p0 = bu.gmm_tied(z, k, 0); p3 = bu.gmm_tied(0.7 * z + 0.3 * nz, k, 0)
    keep = row < col; i, j = row[keep], col[keep]
    use = valid[i] & valid[j]; i, j = i[use], j[use]
    same = ref[i] == ref[j]
    w0 = (p0[i] == p0[j]) != same; w3 = (p3[i] == p3[j]) != same
    return dict(xy=xy, ref=ref, valid=valid, labels=labels, p0=p0, p3=p3, i=i, j=j, fixed=w0 & ~w3, broke=~w0 & w3,
                ari0=adjusted_rand_score(ref[valid], p0[valid]), ari3=adjusted_rand_score(ref[valid], p3[valid]))


def faint(c, f=0.62):
    return tuple(np.array(to_rgb(c)) * (1 - f) + f)


def edge_map(ax, R, ttl, lt):
    spatial(ax, R["xy"], [faint(c) for c in label_colors(R["labels"].values)], fig)
    for mask, c in [(R["fixed"], GOOD), (R["broke"], BAD)]:
        seg = np.stack([R["xy"][R["i"][mask]], R["xy"][R["j"][mask]]], 1)
        ax.add_collection(LineCollection(seg, colors=c, linewidths=0.9, zorder=5))
    ptitle(ax, lt, ttl, fontsize=FS)


Rp = recompute(next(s for s in GS.index if s.endswith("H1_2")))
Rd = recompute(next(s for s in GS.index if s.endswith("151673")))
CLS = ["Benign", "GG1", "Stroma"]
row4 = []
for i, (key, lt, ttl) in enumerate([("p0", "j", "H1_2, α = 0"), ("p3", "k", "H1_2, α = 0.3")]):
    ax = fig.add_subplot(gs[3, i * 3:(i + 1) * 3]); row4.append(ax)
    spatial(ax, Rp["xy"], match_colors(Rp[key], Rp["labels"].values, CLS), fig)
    ptitle(ax, lt, f"{ttl}  ·  ARI {Rp['ari0' if key == 'p0' else 'ari3']:.2f}", fontsize=FS)
ax = fig.add_subplot(gs[3, 6:9]); row4.append(ax); edge_map(ax, Rp, "H1_2: pairs that changed", "l")
ax = fig.add_subplot(gs[3, 9:12]); row4.append(ax); edge_map(ax, Rd, "DLPFC 151673, α 0 → 0.3", "m")
top_align(row4)
row_notes(fig, row4, ["", "", f"{int(Rp['fixed'].sum())} corrected · {int(Rp['broke'].sum())} newly wrong",
                      f"{int(Rd['fixed'].sum()):,} corrected · {int(Rd['broke'].sum())} newly wrong"])
maps_legend(fig, row4, cls_handles(CLS) + [Line2D([], [], color=GOOD, lw=1.6, label="pair corrected"),
                                           Line2D([], [], color=BAD, lw=1.6, label="pair newly wrong")], gap=0.022)
print("H1_2 ARI", round(Rp["ari0"], 3), round(Rp["ari3"], 3), "151673 ARI", round(Rd["ari0"], 3), round(Rd["ari3"], 3))
top_legend(fig, [Line2D([], [], marker="o", ls="", ms=5, mfc=DCOL[d], mec=SURF, label={"DLPFC": "DLPFC", "prostate": "Prostate cancer",
                                                                                          "breast": "Breast cancer"}[d]) for d in DCOL]
           + cat_handles(), y=0.94, ncol=6, cs=0.9)
msave(fig, "Figure4")
print("M4_DONE")
