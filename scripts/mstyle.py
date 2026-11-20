"""mstyle.py - style of the six main figures (larger type than the full figures, which become Figures S1-S12).

Page: 180 mm wide (7.09 in), at most 240 mm high. Type: 7 pt text and axis labels, 6.5 pt ticks and legends,
7.5 pt bold panel titles; nothing smaller than 6 pt.
"""
from fig_common import *            # noqa: F401,F403  palette, helpers, data access
from matplotlib.gridspec import GridSpec  # noqa: F401

FW = 7.09
FS, FT, FTK, FMIN = 7.0, 7.5, 6.5, 6.0
plt.rcParams.update({"font.family": ["Nimbus Sans", "DejaVu Sans"], "font.size": FS, "axes.titlesize": FT, "axes.labelsize": FS, "xtick.labelsize": FTK,
                     "ytick.labelsize": FTK, "legend.fontsize": FTK, "axes.titlepad": 4, "axes.labelpad": 2,
                     "axes.linewidth": 0.7, "lines.linewidth": 1.1})
FIGM = os.path.join(bu.BASE, "results/figures_main"); os.makedirs(FIGM, exist_ok=True)


def msave(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGM, f"{name}.{ext}"), bbox_inches="tight", facecolor=SURF, pad_inches=0.05)
    plt.close(fig); print("saved", name, flush=True)


def ptitle(ax, letter, text, **kw):
    """panel title: bold letter, then the short title"""
    kw.setdefault("loc", "left"); kw.setdefault("fontsize", FT)
    ax.set_title(f"{letter}   {text}", **kw)


def method_rows_m(ax, order, labels=True, fs=FTK):
    method_rows(ax, order, labels, fs=fs)


def strip_m(ax, values_by_method, order, rng, xlab=None, labels=True):
    strip(ax, values_by_method, order, rng, s_small=7, s_big=26, xlab=xlab, labels=labels)
    ax.tick_params(axis="y", labelsize=FTK)


def maps_legend(fig, axes, handles, gap=0.006, ncol=None):
    return legend_below(fig, axes, handles, ncol=ncol, gap=gap, fs=FTK)


def top_legend(fig, handles, y=0.995, ncol=None, cs=1.4):
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
                      fontsize=FTK, handletextpad=0.3, columnspacing=cs)


def row_notes(fig, axes, texts, gap=0.004):
    """one caption line under each map of a row, all on the baseline of the lowest map"""
    fig.canvas.draw()
    y0 = min(a.get_position().y0 for a in axes)
    for a, t in zip(axes, texts):
        p = a.get_position()
        fig.text((p.x0 + p.x1) / 2, y0 - gap, t, ha="center", va="top", fontsize=FMIN, color=INK2)


def top_align(axes):
    for a in axes:
        a.set_anchor("N")


def cat_handles(datasets=()):
    h = [Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec=SURF, label=l) for c, l in
         [(C_SPATIAL, LEG["spatial"]), (C_SIMPLE, LEG["simple"]), (C_NONSP, LEG["nonspatial"])]]
    return h + [Line2D([], [], marker="o", ls="", ms=5, mfc=DSCOL.get(d, MAGENTA), mec=SURF, label=DSNAME.get(d, d))
                for d in datasets]


def cls_handles(classes, extra=()):
    h = [Line2D([], [], marker="o", ls="", ms=5, mfc=CLASS_COL.get(c, MUTED), mec=SURF, mew=0.5, label=CLASS_NAME.get(c, c))
         for c in classes]
    return h + [Line2D([], [], marker="o", ls="", ms=5, mfc=c, mec=AXIS, mew=0.4, label=l) for c, l in extra]


def pick_seed(runs, method):
    """seed whose ARI is closest to the method's mean on the section (a typical, not the best, run)"""
    rr = runs[runs["method"] == method]
    return int(rr.iloc[(rr["ari"] - rr["ari"].mean()).abs().argsort().iloc[0]]["seed"]), int(rr["k"].iloc[0]), rr["ari"].mean()
