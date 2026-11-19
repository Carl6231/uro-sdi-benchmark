"""fig_common.py - shared style, palette and helpers for the full multi-panel figures (Fig 1-7, panels a-o).
Palette: reference categorical slots (light mode); method categories use slots 1-3, dataset emphasis = violet accent
(prostate) + de-emphasis gray (DLPFC); cortical layers take slots 1-7 in order so that touching layers are adjacent
slots; text is always in ink tokens.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

FIG = os.path.join(bu.BASE, "results/figures_full"); os.makedirs(FIG, exist_ok=True)
TAB = os.path.join(bu.BASE, "results/tables")
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                          "#008300", "#6250d6", "#e34948")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]
DIV_NEG, DIV_MID, DIV_POS = RED, "#f0efec", BLUE           # diverging: red (worse) - gray - blue (better)
C_SPATIAL, C_SIMPLE, C_NONSP = BLUE, AQUA, ORANGE
DSCOL = {"prostate": VIOLET, "DLPFC": MUTED, "ccRCC": INK}
DSNAME = {"prostate": "Prostate cancer", "DLPFC": "DLPFC", "ccRCC": "ccRCC"}
NAME = {"graphst": "GraphST", "smooth": "PCA-smooth", "stagate": "STAGATE", "sedr": "SEDR", "drsc": "DR-SC",
        "bayesspace": "BayesSpace", "banksy": "BANKSY", "gmm": "GMM", "spagcn": "SpaGCN", "leiden": "Leiden",
        "spaceflow": "SpaceFlow", "spagcn_refined": "SpaGCN (refined)", "bass": "BASS", "precast": "PRECAST"}
METHODS = ["graphst", "smooth", "stagate", "sedr", "drsc", "bayesspace", "banksy", "gmm", "spagcn", "leiden", "spaceflow",
           "bass", "precast"]
# all spatial methods (10 published + PCA-smooth); the name SP9 is kept from the first version of the benchmark
SP9 = ["graphst", "stagate", "sedr", "spagcn", "spaceflow", "bayesspace", "banksy", "drsc", "smooth", "bass", "precast"]
NSP = len(SP9)
CAT = {m: "spatial" for m in NAME}; CAT["smooth"] = "simple"; CAT["gmm"] = CAT["leiden"] = "nonspatial"
COL = {m: {"spatial": C_SPATIAL, "simple": C_SIMPLE, "nonspatial": C_NONSP}[CAT[m]] for m in NAME}
LEG = {"spatial": "Published spatial method", "simple": "Simple spatial baseline (PCA-smooth)", "nonspatial": "Non-spatial baseline"}
CLASS_COL = {"L1": BLUE, "L2": ORANGE, "L3": AQUA, "L4": YELLOW, "L5": MAGENTA, "L6": GREEN, "WM": VIOLET,
             "Stroma": "#d6d4cb", "Benign": AQUA, "Benign*": GREEN, "GG1": YELLOW, "GG2": ORANGE, "GG4": RED,
             "GG4 Cribriform": MAGENTA, "PIN": VIOLET, "Transition_State": BLUE, "Nerve": "#184f95", "Fat": "#9ec5f4",
             "Inflammation": INK2, "TLS": VIOLET, "NO_TLS": "#e6e5df", "T_agg": ORANGE}
CLASS_NAME = {"GG1": "Gleason GG1", "GG2": "Gleason GG2", "GG4": "Gleason GG4", "GG4 Cribriform": "GG4 cribriform",
              "Benign*": "Benign (other)", "Transition_State": "Transition state", "NO_TLS": "No TLS", "T_agg": "T-cell aggregate"}
UNLABELLED = "#f3f2ee"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.2, "axes.titlesize": 6.8, "axes.labelsize": 6.0, "axes.titleweight": "bold",
    "xtick.labelsize": 5.5, "ytick.labelsize": 5.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2, "text.color": INK, "axes.titlecolor": INK,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 2, "ytick.major.pad": 2, "legend.frameon": False,
    "legend.fontsize": 5.8, "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURF,
    "figure.facecolor": SURF, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titlelocation": "left", "axes.titlepad": 5,
    "axes.labelpad": 2})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", facecolor=SURF, pad_inches=0.04)
    plt.close(fig); print("saved", name, flush=True)


def title(ax, letter, text, **kw):
    ax.set_title(f"{letter}   {text}", **kw)


def gridx(ax):
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)


def gridy(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)


def zero(ax, axis="y"):
    (ax.axhline if axis == "y" else ax.axvline)(0, color=MUTED, lw=0.7, zorder=1)


def short(sid):
    """readable section id: 151673, H1_2, GSM5924042"""
    if "Maynard" in sid: return sid.split("_")[-1]
    if "Erickson" in sid: return "_".join(sid.split("_")[-2:])
    return sid.split("_")[1] if sid.startswith("GSE") else sid


def scores(path="results/scores.csv"):
    S = pd.read_csv(os.path.join(bu.BASE, path))
    return S[S["status"] == "ok"].copy()


def load_section(sid):
    """reference labels (str or NaN) and coordinates (DataFrame x, y) of every spot in a section"""
    a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), backed="r")
    obs = a.obs.copy(); xy = pd.DataFrame(np.asarray(a.obsm["spatial"], float), index=obs.index, columns=["x", "y"]); a.file.close()
    lab = obs["label"].astype(object) if "label" in obs else pd.Series(np.nan, index=obs.index, dtype=object)
    lab = lab.where(~lab.astype(str).isin(bu.EXCLUDE_LABELS)).where(lab.notna())
    return lab.map(lambda v: v if pd.isna(v) else str(v)), xy


def run_labels(sid, method, k, seed, runs_dir=None):
    f = os.path.join(runs_dir or bu.RUNS, sid, f"{method}_k{k}_s{seed}.csv")
    return pd.read_csv(f).set_index("spot")["cluster"].astype(int)


def match_colors(pred, lab, classes, other=("#5c5b57", "#a3a29c", "#c9c8c1", "#3d3c39")):
    """colour clusters by the reference class they match best (Hungarian assignment on the contingency table)"""
    pred = pd.Series(pred); lab = pd.Series(lab, index=pred.index)
    ok = lab.notna().values
    cl = np.unique(pred.values)
    C = np.array([[np.sum((pred.values[ok] == c) & (lab.values[ok] == k)) for k in classes] for c in cl])
    r, c = linear_sum_assignment(-C)
    mp = {cl[i]: CLASS_COL.get(classes[j], MUTED) for i, j in zip(r, c)}
    rest = [x for x in cl if x not in mp]
    for i, x in enumerate(rest):
        mp[x] = other[i % len(other)]
    return [mp[v] for v in pred.values]


def spatial(ax, xy, colors, fig, fill=0.98, edge=None):
    """draw Visium spots as touching discs (size from the spot spacing and the axes scale)"""
    xy = np.asarray(xy, float)
    nn = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    ax.set_xlim(xy[:, 0].min() - nn, xy[:, 0].max() + nn); ax.set_ylim(xy[:, 1].max() + nn, xy[:, 1].min() - nn)
    ax.set_aspect("equal"); ax.axis("off")
    fig.canvas.draw()
    bb = ax.get_window_extent()
    sx = bb.width / (ax.get_xlim()[1] - ax.get_xlim()[0]); sy = bb.height / abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    d = fill * nn * min(sx, sy) * 72 / fig.dpi
    ax.scatter(xy[:, 0], xy[:, 1], s=d ** 2, c=colors, marker="o", linewidths=0 if edge is None else 0.25,
               edgecolors=edge or "none")
    return nn


def label_colors(lab):
    return [CLASS_COL.get(v, MUTED) if isinstance(v, str) else UNLABELLED for v in lab]


def direct_labels(ax, xy, lab, classes, fs=5.4):
    """class names written on the map on a member spot of each class (white halo), kept apart from each other"""
    xy = np.asarray(xy, float); lab = np.asarray(lab, dtype=object)
    nn = np.median(cKDTree(xy).query(xy, k=2)[0][:, 1])
    placed = []
    for k in classes:
        m = lab == k
        if m.sum() < 5:
            continue
        p = xy[m]; c = np.median(p, 0)
        cand = p[np.argsort(((p - c) ** 2).sum(1))][:max(5, int(0.6 * len(p)))]
        txt = CLASS_NAME.get(k, k); wx = (len(txt) * 0.62 + 2) * nn; wy = 3.2 * nn      # rough label extent in data units
        def clash(q):
            return any(abs(q[0] - r[0]) < (wx + r[2]) / 2 and abs(q[1] - r[1]) < (wy + r[3]) / 2 for r in placed)
        q = next((q for q in cand if not clash(q)), cand[0])
        placed.append((q[0], q[1], wx, wy))
        ax.text(q[0], q[1], txt, fontsize=fs, ha="center", va="center", color=INK,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")], zorder=5)


def place_labels(ax, fig, xs, ys, texts, fs=5.2, avoid=(), color=INK):
    """greedy label placement: no label covers a data point, another label or leaves the axes"""
    fig.canvas.draw(); rend = fig.canvas.get_renderer()
    pts = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    placed = [a.get_window_extent(rend).expanded(1.03, 1.08) for a in avoid]
    axbb = ax.get_window_extent(rend)
    near = [(4, 0), (-4, 0), (0, 5), (0, -5), (4, 4), (-4, 4), (4, -4), (-4, -4), (8, 0), (-8, 0), (0, 9), (0, -9),
            (9, 7), (-9, 7), (9, -7), (-9, -7)]
    far = [(16, 10), (-16, 10), (16, -10), (-16, -10), (20, 0), (-20, 0), (0, 16), (0, -16), (22, 14), (-22, 14), (22, -14), (-22, -14)]
    def ok(bb):
        return not (any(bb.overlaps(o) for o in placed) or any(bb.contains(px, py) for px, py in pts)
                    or bb.x0 < axbb.x0 or bb.x1 > axbb.x1 or bb.y0 < axbb.y0 or bb.y1 > axbb.y1)
    for x, y, t in zip(xs, ys, texts):
        done = False
        for dx, dy in near + far:
            ha = "left" if dx > 0 else ("right" if dx < 0 else "center"); va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
            kw = dict(arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4, shrinkA=0, shrinkB=2)) if (dx, dy) in far else {}
            a = ax.annotate(t, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=fs, color=color, **kw)
            bb = a.get_window_extent(rend).expanded(1.03, 1.08)
            if ok(bb):
                placed.append(bb); done = True; break
            a.remove()
        if not done:
            print("label not placed cleanly:", t)
            ax.annotate(t, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=fs, color=color, va="center")


def legend_below(fig, axes, handles, ncol=None, gap=0.012, fs=5.8, **kw):
    """legend centred under a group of axes (after the layout is fixed)"""
    fig.canvas.draw()
    bbs = [a.get_position() for a in axes]
    x0 = min(b.x0 for b in bbs); x1 = max(b.x1 for b in bbs); y0 = min(b.y0 for b in bbs)
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=((x0 + x1) / 2, y0 - gap), ncol=ncol or len(handles),
                      fontsize=fs, handletextpad=0.3, columnspacing=1.2, **kw)


def class_handles(classes):
    return [Line2D([], [], marker="o", ls="", ms=4.5, mfc=CLASS_COL.get(c, MUTED), mec=SURF, mew=0.5, label=CLASS_NAME.get(c, c))
            for c in classes]


def category_handles(datasets=()):
    h = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=c, mec=SURF, label=l) for c, l in
         [(C_SPATIAL, LEG["spatial"]), (C_SIMPLE, LEG["simple"]), (C_NONSP, LEG["nonspatial"])]]
    return h + [Line2D([], [], marker="o", ls="", ms=4.5, mfc=DSCOL[d], mec=SURF, label=DSNAME[d]) for d in datasets]


def method_rows(ax, order, labels=True, fs=5.4):
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAME[m] for m in order][::-1] if labels else [], fontsize=fs)
    ax.set_ylim(-0.6, len(order) - 0.4)


def class_legend(ax_or_fig, classes, loc="upper center", anchor=None, ncol=None, fs=5.8, **kw):
    h = [Line2D([], [], marker="o", ls="", ms=4.5, mfc=CLASS_COL.get(c, MUTED), mec=SURF, mew=0.5, label=CLASS_NAME.get(c, c))
         for c in classes]
    return ax_or_fig.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=ncol or len(h), fontsize=fs, handletextpad=0.2,
                            columnspacing=0.8, **kw)


def category_legend(fig_or_ax, loc="upper center", anchor=None, ncol=3, cats=("spatial", "simple", "nonspatial"), **kw):
    h = [Line2D([], [], marker="o", ls="", ms=4.5, mfc={"spatial": C_SPATIAL, "simple": C_SIMPLE, "nonspatial": C_NONSP}[c],
                mec=SURF, mew=0.6, label=LEG[c]) for c in cats]
    return fig_or_ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=ncol, handletextpad=0.2, columnspacing=1.2, **kw)


def dataset_legend(fig_or_ax, datasets=("DLPFC", "prostate"), loc="upper center", anchor=None, n=None, **kw):
    h = [Line2D([], [], color=DSCOL[d], lw=1.6, marker="o", ms=3.5, mec=SURF,
                label=DSNAME[d] + (f" ({n[d]} sections)" if n else "")) for d in datasets]
    return fig_or_ax.legend(handles=h, loc=loc, bbox_to_anchor=anchor, ncol=len(h), columnspacing=1.6, **kw)


def strip(ax, values_by_method, order, rng, s_small=6, s_big=24, xlab=None, labels=True):
    """per-section dots (light) and mean (large) for each method, one row per method, first method at the top"""
    for i, m in enumerate(order):
        y = len(order) - 1 - i
        v = np.asarray(values_by_method.get(m, []), float); v = v[~np.isnan(v)]
        if len(v) == 0:
            continue
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=s_small, color=COL[m], alpha=0.45, lw=0, zorder=2)
        ax.scatter([v.mean()], [y], s=s_big, color=COL[m], edgecolor=SURF, lw=0.8, zorder=3)
    method_rows(ax, order, labels)
    if xlab:
        ax.set_xlabel(xlab)


def heat(ax, M, cmap, vmin, vmax, fmt="{:.2f}", fs=4.6, text=True, xt=None, yt=None, dark=0.6):
    """annotated heatmap; text switches to white on dark cells"""
    from matplotlib.colors import LinearSegmentedColormap
    cm = cmap if not isinstance(cmap, list) else LinearSegmentedColormap.from_list("c", cmap)
    A = np.asarray(M, float)
    ax.imshow(A, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    if text:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if np.isnan(A[i, j]):
                    continue
                f = (A[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", fontsize=fs,
                        color=SURF if (f > dark if cmap is not None else False) else INK)
    ax.set_xticks(range(A.shape[1])); ax.set_yticks(range(A.shape[0]))
    if xt is not None: ax.set_xticklabels(xt)
    if yt is not None: ax.set_yticklabels(yt)
    for s in ax.spines.values():
        s.set_visible(False)
    return cm


def align_titles(fig, axes):
    """Put the left titles of one row of panels on a common baseline (the highest one in the row)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y = max(a._left_title.get_window_extent(r).y0 for a in axes)
    for a in axes:
        t = a._left_title
        ya = a.transAxes.inverted().transform((0, y))[1]
        a.set_title(t.get_text(), loc="left", y=ya, pad=0, va="bottom", fontsize=t.get_fontsize(), fontweight=t.get_fontweight())


def boot_ci(x, B=5000, seed=1):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), size=(B, len(x)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)
