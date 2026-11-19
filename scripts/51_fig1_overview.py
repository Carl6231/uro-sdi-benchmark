"""51_fig1_overview.py - Figure 1: study design, data and workload (panels a-o)."""
from fig_common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

S = scores()
main = S[S["method"] != "spagcn_refined"]
secs = {ds: sorted(main.loc[main["dataset"] == ds, "slide"].unique()) for ds in ["DLPFC", "prostate", "ccRCC"]}
tls_secs = sorted(S.loc[S["tls_f1"].notna(), "slide"].unique())
n_other = main.loc[main["dataset"] == "other", "slide"].nunique()
D = pd.read_csv(os.path.join(bu.BASE, "results/dose/smooth_dose.csv"))
SD = scores("results/scores_dose.csv"); KS = scores("results/scores_ksens.csv"); KS = KS[KS["method"] != "spagcn_refined"]
n_fixedk = 19 * 3 * 3

info, comp = [], {}
for ds in ["DLPFC", "prostate", "ccRCC"]:
    for sid in secs[ds]:
        a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"))
        tot = np.asarray(a.X.sum(1)).ravel()
        lab, _ = load_section(sid); vc = lab.value_counts()
        info.append(dict(slide=sid, dataset=ds, n_spots=a.n_obs, median_umi=float(np.median(tot)),
                         n_classes=int((vc >= bu.MIN_CLASS_SPOTS).sum()), n_tls=int(vc.get("TLS", 0)), n_lab=int(lab.notna().sum())))
        comp[sid] = vc / vc.sum()
I = pd.DataFrame(info).set_index("slide")
I.to_csv(os.path.join(TAB, "section_overview.csv"))

fig = plt.figure(figsize=(7.4, 11.3))
gs = GridSpec(7, 12, figure=fig, height_ratios=[0.95, 1.8, 0.95, 1.5, 0.7, 1.55, 1.2], hspace=0.62, wspace=2.6)
rng = np.random.default_rng(0)

# a: study design
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
n_main = len(main)
boxes = [("Public Visium data", f"{len(secs['DLPFC'])} DLPFC (layers)\n{len(secs['prostate'])} prostate (Gleason)\n"
          f"{len(secs['ccRCC'])} ccRCC ({len(tls_secs)} with TLS)\n+{n_other} unlabelled"),
         ("Pre-processing", "QC, 3,000 HVGs, PCA\nK = annotated classes\n(>= 20 spots)"),
         (f"{len(METHODS)} methods", f"{len(METHODS) - 3} published spatial\nPCA-smooth baseline\nGMM, Leiden"),
         ("Benchmark", f"{n_main:,} clusterings\n1-5 seeds per section\nall runs completed"),
         ("Metrics", "ARI, NMI, AMI\nhomogeneity\ntumour/TLS F1, PAS"),
         ("Follow-up", "label geometry\ndose-response\nK sweep, bootstrap")]
w, gap = 0.148, 0.019
for i, (h, t) in enumerate(boxes):
    x0 = 0.008 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x0, 0.04), w, 0.80, boxstyle="round,pad=0.006,rounding_size=0.02", fc="#f3f2ee", ec=AXIS, lw=0.7,
                                clip_on=False))
    ax.text(x0 + w / 2, 0.72, h, ha="center", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(x0 + w / 2, 0.34, t, ha="center", va="center", fontsize=5.0, color=INK2, linespacing=1.3)
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x0 + w + 0.002, 0.44), (x0 + w + gap - 0.002, 0.44), arrowstyle="-|>", mutation_scale=6,
                                     color=INK2, lw=0.7))
ax.set_title("a   Study design", loc="left")

# b-d: reference annotations
tls_best = I.loc[tls_secs, "n_tls"].idxmax()
maps = []
for col, (sid, lt, txt) in enumerate([(next(s for s in secs["DLPFC"] if s.endswith("151673")), "b", "DLPFC 151673: layers"),
                                       (next(s for s in secs["prostate"] if s.endswith("H1_2")), "c", "Prostate H1_2: Gleason"),
                                       (tls_best, "d", f"ccRCC {short(tls_best)}: TLS")]):
    ax = fig.add_subplot(gs[1, col * 4:(col + 1) * 4]); maps.append(ax)
    lab, xy = load_section(sid)
    spatial(ax, xy.values, label_colors(lab.values), fig)
    vc = lab.value_counts()
    direct_labels(ax, xy.values, lab.values, [c for c in vc.index if vc[c] >= 20], fs=5.0)
    ax.set_title(f"{lt}   {txt}", loc="left")

# e-g: section size, classes, sequencing depth (shared dataset rows)
rows = ["ccRCC", "prostate", "DLPFC"]
for c0, lt, col_, xl, tt in [(0, "e", "n_spots", "Spots per section", "Section size"),
                             (4, "f", "n_classes", "Classes with >= 20 spots", "Annotated classes"),
                             (8, "g", "median_umi", "Median UMIs per spot", "Sequencing depth")]:
    ax = fig.add_subplot(gs[2, c0:c0 + 4]); gridx(ax)
    for yy, ds in enumerate(rows):
        v = I.loc[I["dataset"] == ds, col_].values
        ax.scatter(v, yy + rng.uniform(-0.16, 0.16, len(v)), s=8, color=DSCOL[ds], alpha=0.75, lw=0, zorder=3)
        ax.plot([np.median(v)] * 2, [yy - 0.3, yy + 0.3], color=INK, lw=1.0, zorder=4)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([f"ccRCC ({len(secs['ccRCC'])})", f"Prostate ({len(secs['prostate'])})", f"DLPFC ({len(secs['DLPFC'])})"] if c0 == 0 else [])
    ax.set_ylim(-0.5, 2.5); ax.set_xlabel(xl); ax.set_title(f"{lt}   {tt}", loc="left")
    if col_ == "n_classes": ax.set_xticks(range(0, int(I["n_classes"].max()) + 1))
    if col_ == "median_umi":
        from matplotlib.ticker import LogLocator, FuncFormatter, NullLocator
        ax.set_xscale("log"); ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k")); ax.xaxis.set_minor_locator(NullLocator())

# h, i: class composition; j: TLS prevalence
p_order = ["Stroma", "Benign", "Benign*", "GG1", "GG2", "GG4", "GG4 Cribriform", "PIN", "Transition_State", "Fat", "Nerve", "Inflammation"]
comp_axes = []
for (c0, c1, lt, ds, tt, order) in [(0, 6, "h", "prostate", "Prostate: class composition", p_order),
                                    (6, 9, "i", "DLPFC", "DLPFC: layers", ["L1", "L2", "L3", "L4", "L5", "L6", "WM"])]:
    ax = fig.add_subplot(gs[3, c0:c1]); comp_axes.append(ax)
    seen = []
    for yy, sid in enumerate(secs[ds]):
        left, cc = 0.0, comp[sid]
        for k in order + [k for k in cc.index if k not in order]:
            if k in cc.index and cc[k] > 0:
                ax.barh(yy, cc[k], left=left, color=CLASS_COL.get(k, MUTED), height=0.8, lw=0.3, edgecolor=SURF)
                left += cc[k]
                if k not in seen: seen.append(k)
    ax.set_yticks(range(len(secs[ds]))); ax.set_yticklabels([short(s) for s in secs[ds]], fontsize=5 if ds == "prostate" else 4.6)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Share of labelled spots")
    for s_ in ["left", "bottom"]: ax.spines[s_].set_visible(False)
    ax.set_title(f"{lt}   {tt}", loc="left")
    leg = [k for k in order if k in seen] + [k for k in seen if k not in order]
    ax.legend(handles=class_handles(leg), loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=5 if ds == "prostate" else 4,
              fontsize=4.9, handletextpad=0.1, columnspacing=0.6)
ax = fig.add_subplot(gs[3, 9:12]); gridy(ax)
t = I.loc[I["dataset"] == "ccRCC"].assign(pct=lambda d: d["n_tls"] / d["n_lab"].clip(lower=1) * 100).sort_values("pct", ascending=False)
ax.bar(range(len(t)), t["pct"], color=[VIOLET if s in tls_secs else AXIS for s in t.index], width=0.75)
ax.set_xticks([]); ax.set_xlabel("ccRCC sections, sorted"); ax.set_ylabel("TLS spots (% of labelled)")
ax.legend(handles=[Patch(color=VIOLET, label=f"scored ({len(tls_secs)})"), Patch(color=AXIS, label="not scored*")],
          loc="upper right", fontsize=5)
ax.set_title("j   ccRCC: TLS share", loc="left")

# k-m: methods, runtime, GPU memory (shared method rows)
order_k = ["graphst", "stagate", "sedr", "spaceflow", "spagcn", "bayesspace", "banksy", "drsc", "bass", "precast", "smooth", "gmm", "leiden"]
order_k = [m for m in order_k if m in main["method"].unique()]
gpu = main.groupby("method")["gpu_peak_mb"].median()
F = {"graphst": [1, 1, 1, 0, 0, 1], "stagate": [1, 1, 1, 0, 0, 1], "sedr": [1, 1, 1, 0, 0, 1], "spaceflow": [1, 1, 1, 0, 0, 1],
     "spagcn": [1, 1, 0, 0, 0, 0], "bayesspace": [1, 0, 0, 1, 1, 0], "banksy": [1, 0, 0, 1, 1, 0], "drsc": [1, 0, 0, 1, 1, 0],
     "smooth": [1, 0, 0, 1, 0, 1], "gmm": [0, 0, 0, 1, 0, 1], "leiden": [0, 0, 0, 0, 0, 0],
     "bass": [1, 0, 0, 1, 1, 0], "precast": [1, 0, 0, 1, 1, 0]}
GPU_MB = 25  # every run allocates a ~17 MB shared baseline; above this the method itself uses the GPU
for m in F: F[m][2] = int(gpu.get(m, 0) > GPU_MB)
ax = fig.add_subplot(gs[5, 0:4]); row5 = [ax]
feats = ["spatial", "deep learning", "GPU", "probabilistic", "R package", "shared GMM"]
for i, m in enumerate(order_k):
    for j, v in enumerate(F[m]):
        ax.scatter(j, len(order_k) - 1 - i, s=22 if v else 4, color=COL[m] if v else GRID, lw=0, zorder=3)
ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=40, ha="left", fontsize=5.2); ax.xaxis.tick_top()
method_rows(ax, order_k); ax.set_xlim(-0.6, len(feats) - 0.4)
for s_ in ax.spines.values(): s_.set_visible(False)
ax.set_title("k   Methods compared", loc="left", pad=3)
ax = fig.add_subplot(gs[5, 4:8]); gridx(ax); row5.append(ax)
rt = main.groupby("method")["sec"].describe(percentiles=[0.25, 0.5, 0.75])
for i, m in enumerate(order_k):
    y = len(order_k) - 1 - i
    ax.plot([rt.loc[m, "25%"], rt.loc[m, "75%"]], [y, y], color=COL[m], lw=1.6, solid_capstyle="round")
    ax.scatter(rt.loc[m, "50%"], y, s=16, color=COL[m], edgecolor=SURF, lw=0.6, zorder=3)
ax.set_xscale("log"); method_rows(ax, order_k, labels=False)
ax.set_xlabel("Seconds per run (median, IQR)"); ax.set_title("l   Runtime", loc="left")
ax = fig.add_subplot(gs[5, 8:12]); gridx(ax); row5.append(ax)
gm_ = main.groupby("method")["gpu_peak_mb"].agg(["median", "max"])
for i, m in enumerate(order_k):
    y = len(order_k) - 1 - i
    if gm_.loc[m, "median"] > GPU_MB:
        ax.barh(y, gm_.loc[m, "median"], color=COL[m], height=0.6)
        ax.plot([gm_.loc[m, "max"]] * 2, [y - 0.3, y + 0.3], color=INK2, lw=0.8)
    else:
        ax.text(15, y, "CPU only", va="center", fontsize=4.8, color=MUTED)
method_rows(ax, order_k, labels=False)
ax.set_xlabel("Peak GPU memory (MB)"); ax.set_title("m   GPU memory", loc="left")

# n: runs per experiment and method; o: totals
ax = fig.add_subplot(gs[6, 0:8])
dose_m = SD["method"].str.split("-").str[0]
rows_n = ["DLPFC", "Prostate", "ccRCC", "Unlabelled", "Dose sweeps", "K sweep"]
M = np.array([[int(((main["method"] == m) & (main["dataset"] == ds_)).sum()) for m in order_k] for ds_ in ["DLPFC", "prostate", "ccRCC", "other"]] +
             [[int((dose_m == m).sum()) + (len(D) if m == "smooth" else 0) for m in order_k], [int((KS["method"] == m).sum()) for m in order_k]], float)
heat(ax, np.log10(M + 1), SEQ[:9], 0, np.log10(M.max() + 1), text=False, xt=[NAME[m] for m in order_k], yt=rows_n)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, f"{int(v):,}" if v else "–", ha="center", va="center", fontsize=4.8,
                color=SURF if np.log10(v + 1) / np.log10(M.max() + 1) > 0.62 else INK)
ax.tick_params(axis="x", rotation=35, labelsize=5.2)
ax.set_title("n   Runs per experiment and method", loc="left")
ax = fig.add_subplot(gs[6, 9:12]); gridy(ax)
tot = pd.Series({"Benchmark": n_main, "α sweep": len(D), "λ sweep": int((dose_m == "banksy").sum()), "radius": int((dose_m == "stagate").sum()),
                 "K sweep": len(KS), "fixed K": n_fixedk})
ax.bar(range(len(tot)), tot.values, color=[SEQ[7]] + [SEQ[4]] * (len(tot) - 1), width=0.65)
for i, v in enumerate(tot.values):
    ax.text(i, v * 1.15, f"{v:,}", ha="center", fontsize=4.8, color=INK2)
ax.set_yscale("log"); ax.set_ylim(50, tot.max() * 5)
ax.set_xticks(range(len(tot))); ax.set_xticklabels(tot.index, rotation=35, ha="right", fontsize=5.2)
ax.set_ylabel("Runs"); ax.set_title(f"o   {tot.sum():,} runs in total", loc="left")
align_titles(fig, row5)
save(fig, "Fig1_design_data_workload")
print("FIG1_DONE")
