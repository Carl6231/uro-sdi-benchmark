"""53_fig3_transfer_maps.py - Figure 3: does the DLPFC ranking transfer to prostate cancer? + domain maps of all 11 methods."""
from fig_common import *
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr

S = scores(); main = S[S["method"].isin(METHODS)]
P = {ds: main[main["dataset"] == ds].groupby(["method", "slide"])["ari"].mean().unstack() for ds in ["DLPFC", "prostate"]}
R = pd.DataFrame({ds: P[ds].mean(1) for ds in P})
fig = plt.figure(figsize=(7.4, 8.2))
gs = GridSpec(4, 12, figure=fig, height_ratios=[1.75, 1.2, 1.2, 1.2], hspace=0.32, wspace=0.9)

# a: method means in both tissues
ax = fig.add_subplot(gs[0, 0:4]); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
rho, p = spearmanr(R["DLPFC"], R["prostate"])
for m, r in R.iterrows():
    ax.scatter(r["DLPFC"], r["prostate"], s=24, color=COL[m], edgecolor=SURF, lw=0.8, zorder=3)
ax.set_xlim(R["DLPFC"].min() - 0.04, R["DLPFC"].max() + 0.04); ax.set_ylim(R["prostate"].min() - 0.03, R["prostate"].max() + 0.035)
ax.set_xlabel("Mean ARI, DLPFC"); ax.set_ylabel("Mean ARI, prostate")
ax.set_title(f"a   Means in two tissues (ρ = {rho:.2f}, P = {p:.2f})", loc="left")
place_labels(ax, fig, R["DLPFC"].values, R["prostate"].values, [NAME[m] for m in R.index], fs=5.0)

# b: rank change
ax = fig.add_subplot(gs[0, 5:8])
rk1 = R["DLPFC"].rank(ascending=False); rk2 = R["prostate"].rank(ascending=False); n = len(R)
for m in R.index:
    ax.plot([0, 1], [rk1[m], rk2[m]], color=COL[m], lw=1.0, zorder=2)
    ax.scatter([0, 1], [rk1[m], rk2[m]], s=12, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3)
    ax.text(-0.08, rk1[m], f"{NAME[m]} {int(rk1[m])}", ha="right", va="center", fontsize=5)
    ax.text(1.08, rk2[m], f"{int(rk2[m])} {NAME[m]}", ha="left", va="center", fontsize=5)
ax.set_ylim(n + 0.6, 0.4); ax.set_xlim(-0.04, 1.04); ax.set_xticks([0, 1]); ax.set_xticklabels(["DLPFC", "Prostate"])
ax.set_yticks([]); [ax.spines[s].set_visible(False) for s in ["left", "bottom"]]
ax.set_title("b   Rank change", loc="center")

# c: pairwise wins on the prostate sections
ax = fig.add_subplot(gs[0, 9:12])
o = R["prostate"].sort_values(ascending=False).index.tolist(); Pp = P["prostate"].loc[o]
W = np.array([[np.sum(Pp.loc[a_].values > Pp.loc[b_].values) if a_ != b_ else np.nan for b_ in o] for a_ in o])
heat(ax, W, SEQ[:11], 0, Pp.shape[1], fmt="{:.0f}", fs=4.4, xt=[str(i + 1) for i in range(len(o))],
     yt=[f"{i + 1} {NAME[m]}" for i, m in enumerate(o)])
ax.yaxis.tick_right(); ax.tick_params(axis="x", labelsize=4.8); ax.tick_params(axis="y", labelsize=4.8)
ax.set_xlabel(f"Column method (prostate rank); cell = sections won of {Pp.shape[1]}", fontsize=4.9)
ax.set_title("c   Head-to-head, prostate", loc="left")

# d-o: domain maps of one prostate section
sid = next(s for s in P["prostate"].columns if s.endswith("H1_2"))
lab, xy_all = load_section(sid)
CLS = ["Benign", "GG1", "Stroma"]
runs = main[main["slide"] == sid]
order_m = runs.groupby("method")["ari"].mean().sort_values(ascending=False).index.tolist()
if len(order_m) > 11:          # 12 map slots: the annotation + the 11 best-scoring methods on this section (rest in the caption)
    keep = order_m[:10] + (["gmm"] if "gmm" not in order_m[:10] else order_m[10:11])
    order_m = [m for m in order_m if m in keep]
panels = [("d", f"{short(sid)}: annotation", None)]
for i, m in enumerate(order_m):
    rr = runs[runs["method"] == m]
    seed = int(rr.iloc[(rr["ari"] - rr["ari"].mean()).abs().argsort().iloc[0]]["seed"])
    panels.append(("efghijklmno"[i], f"{NAME[m]}, ARI {rr['ari'].mean():.2f}", run_labels(sid, m, int(rr["k"].iloc[0]), seed)))
spots = panels[1][2].index
xy = xy_all.loc[spots].values; labs = lab.reindex(spots)
map_axes = []
for i, (lt, tt, cl) in enumerate(panels):
    ax = fig.add_subplot(gs[1 + i // 4, (i % 4) * 3:(i % 4) * 3 + 3]); map_axes.append(ax)
    if cl is None:
        spatial(ax, xy, label_colors(labs.values), fig); direct_labels(ax, xy, labs.values, CLS, fs=5)
    else:
        spatial(ax, xy, match_colors(cl.reindex(spots).values, labs.values, CLS), fig)
    ax.set_title(f"{lt}   {tt}", loc="left", fontsize=6.2, fontweight="bold" if cl is None else "normal")
legend_below(fig, map_axes[-4:], class_handles(CLS) + [Line2D([], [], marker="o", ls="", ms=4.5, mfc=MUTED, mec=SURF, label="other class (< 20 spots)"),
                                                     Line2D([], [], marker="o", ls="", ms=4.5, mfc=UNLABELLED, mec=AXIS, mew=0.4,
                                                            label="not annotated")], gap=0.006)
fig.legend(handles=category_handles(), loc="lower center", bbox_to_anchor=(0.5, 0.905), ncol=3, fontsize=5.8, handletextpad=0.2)
save(fig, "Fig3_transfer_maps")
print("FIG3_DONE")
