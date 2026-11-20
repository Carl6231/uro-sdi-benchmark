"""Summarise same-seed edge transitions (Tables S16-S18; plotted in Figure 4d-m by m4_fig4.py)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


COUNTS = ["retained_correct", "introduced", "corrected", "persistent_error"]
RATES = ["corrected_rate", "introduced_rate", "net_error_change", "error_rate_before", "error_rate_after"]
RUN = ["dataset", "slide", "seed", "alpha", "edge_type"]


def reconcile(raw, doses):
    """Recover both independently tabulated error margins from the paired counts."""
    if raw.duplicated(RUN + ["class_a", "class_b"]).any():
        raise ValueError("Duplicate class pair in a fitted partition")
    if (raw[COUNTS] < 0).any().any() or not raw[COUNTS].sum(axis=1).eq(raw.n_edges).all():
        raise ValueError("Invalid transition counts")
    if not raw.edge_type.eq(np.where(raw.class_a == raw.class_b, "within", "boundary")).all():
        raise ValueError("Edge type disagrees with reference labels")
    if not np.allclose(doses.ari, doses.archived_ari, atol=1e-7, rtol=0):
        raise ValueError("Dose reconstruction differs from the original ARI")

    margins = []
    for edge_type, denominator, errors in [
        ("within", "within_edges", "false_splits"), ("boundary", "boundary_edges", "false_merges")
    ]:
        margins.append(doses.assign(edge_type=edge_type)[RUN + [denominator, errors]]
                       .rename(columns={denominator: "denominator", errors: "errors"}))
    margins = pd.concat(margins, ignore_index=True)
    baseline = margins[margins.alpha.eq(0)].drop(columns="alpha")
    expected = margins[margins.alpha.isin([0.3, 0.7])].merge(
        baseline, on=["dataset", "slide", "seed", "edge_type"], validate="many_to_one",
        suffixes=("_after", "_before"))
    observed = raw.groupby(RUN, as_index=False)[["n_edges", *COUNTS]].sum()
    check = expected.merge(observed, on=RUN, how="outer", indicator=True, validate="one_to_one")
    valid = (check["_merge"].eq("both") & check.n_edges.eq(check.denominator_before)
             & check.n_edges.eq(check.denominator_after)
             & (check.corrected + check.persistent_error).eq(check.errors_before)
             & (check.introduced + check.persistent_error).eq(check.errors_after))
    if not valid.all():
        raise ValueError(f"Paired counts disagree with the archived margins: {check.loc[~valid, RUN].head()}")


def section_rates(rows, labels):
    keys = ["dataset", "donor", "slide", "alpha", "edge_type", *labels]
    runs = rows.groupby(keys + ["seed"], as_index=False, dropna=False)[["n_edges", *COUNTS]].sum()
    by_section = runs.groupby(keys, dropna=False)
    if (set(runs.seed) != {0, 1, 2} or not by_section.seed.nunique().eq(3).all()
            or not by_section.n_edges.nunique().eq(1).all()):
        raise ValueError("Seeds or reference edges differ within a section and comparison")
    out = by_section[["n_edges", *COUNTS]].mean().reset_index()
    out["n_edges"] = out.n_edges.astype(int)
    out["error_rate_before"] = (out.corrected + out.persistent_error) / out.n_edges
    out["corrected_rate"] = out.corrected / out.n_edges
    out["introduced_rate"] = out.introduced / out.n_edges
    out["error_rate_after"] = (out.introduced + out.persistent_error) / out.n_edges
    out["net_error_change"] = (out.introduced - out.corrected) / out.n_edges
    return out[keys + ["n_edges", *RATES]]


def cohort_rates(sections):
    keys = ["dataset", "alpha", "edge_type"]
    units = sections.assign(unit=sections.donor.fillna(sections.slide))
    coverage = units.groupby(keys).agg(
        n_sections=("slide", "nunique"), n_donors=("donor", "nunique"),
        n_section_edges=("n_edges", "sum"),
        sections_fewer_errors=("net_error_change", lambda x: (x < 0).sum()),
        sections_more_errors=("net_error_change", lambda x: (x > 0).sum()),
        sections_unchanged=("net_error_change", lambda x: (x == 0).sum()))
    coverage["aggregation_unit"] = np.where(coverage.n_donors > 0, "donor", "section")
    coverage["n_donors"] = coverage.n_donors.astype("Int64").replace(0, pd.NA)
    means = units.groupby(keys + ["unit"])[RATES].mean().groupby(keys).mean()
    return means.join(coverage).reset_index()


def figure(cohorts, folder):
    cohorts = cohorts.set_index(["dataset", "alpha", "edge_type"])
    names = ["DLPFC", "prostate", "breast"]
    labels = ["DLPFC", "Prostate", "Breast"]
    colours = ["#23689b", "#b34f2a"]
    ymax = 100 * cohorts[["corrected_rate", "introduced_rate"]].to_numpy().max()
    limit = 5 * np.ceil((ymax + 5) / 5)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.9), sharex=True)
    fig.subplots_adjust(left=.13, right=.98, top=.80, bottom=.18, hspace=.40, wspace=.20)
    for row, weight in enumerate([0.3, 0.7]):
        for col, edge_type in enumerate(["within", "boundary"]):
            ax = axes[row, col]
            panel = "abcd"[2 * row + col]
            heading = "Within one reference label" if edge_type == "within" else "Across reference labels"
            ax.set_title(f"{panel}   {heading}   (α = {weight})", loc="left", fontsize=9.5, fontweight="bold")
            for shift, metric, colour in zip([-.18, .18], ["corrected_rate", "introduced_rate"], colours):
                values = [100 * cohorts.loc[(ds, weight, edge_type), metric] for ds in names]
                bars = ax.barh(np.arange(3) + shift, values, height=.29, color=colour, zorder=2)
                ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
            ax.set(yticks=np.arange(3), yticklabels=labels if col == 0 else [], xlim=(0, limit), ylim=(2.55, -.55))
            ax.xaxis.grid(True, color="#e4e5e5", linewidth=.7, zorder=0)
            ax.spines[["left", "top", "right"]].set_visible(False)
            ax.tick_params(axis="y", length=0, labelsize=9)
            ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Neighbour-pair errors corrected and introduced by smoothing", x=.04, y=.975,
                 ha="left", fontsize=14, fontweight="bold")
    fig.legend(handles=[Patch(color=colours[0], label="Corrected (wrong to right)"),
                        Patch(color=colours[1], label="Introduced (right to wrong)")],
               ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(.03, .92), fontsize=9)
    fig.text(.54, .105, "Share of eligible reference pairs (%)", ha="center", fontsize=10)
    fig.text(.04, .028, "DLPFC (12 sections, 3 donors), prostate (7 sections, 1 patient) and breast (20 sections).\n"
             "Each weight is compared with no smoothing on the same pairs and seeds.", fontsize=8, color="#43474a")
    # quick-look plot only; the published version of these bars is Figure 4d, e (scripts/m4_fig4.py)
    fig.savefig(folder / "paired_changes.png", dpi=300)
    plt.close(fig)


def main(folder):
    raw = pd.read_csv(folder / "paired_edges.csv")
    doses = pd.read_csv(folder / "dose_edges.csv")
    reconcile(raw, doses)
    sections = section_rates(raw, [])
    cohorts = cohort_rates(sections)
    prostate = section_rates(raw[raw.dataset.eq("prostate")], ["class_a", "class_b"])
    pairs = prostate.groupby(["alpha", "edge_type", "class_a", "class_b"])
    interfaces = pairs.agg(n_sections=("slide", "nunique"), n_section_edges=("n_edges", "sum")).join(pairs[RATES].mean())
    sections.to_csv(folder / "paired_sections.csv", index=False)
    cohorts.to_csv(folder / "paired_cohorts.csv", index=False)
    interfaces.to_csv(folder / "prostate_label_pairs.csv")
    figure(cohorts, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    main(parser.parse_args().folder)
