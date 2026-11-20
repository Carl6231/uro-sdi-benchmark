"""Summarize edge errors without treating neighbouring pairs as replicates."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["split_rate", "merge_rate", "ari"]
COHORTS = ["DLPFC", "prostate", "breast"]
COLOURS = {"DLPFC": "#316c93", "prostate": "#ab4e36", "breast": "#42795f"}


def section_summary(raw, key):
    group = ["dataset", "donor", "slide", key]
    for column in ["within_edges", "boundary_edges", "n_common_spots", "k"]:
        if raw.groupby(group, dropna=False)[column].nunique().gt(1).any():
            raise ValueError(f"Inconsistent {column} within a section")
    return raw.groupby(group, as_index=False, dropna=False).agg(
        n_seeds=("seed", "nunique"), k=("k", "first"), n_common_spots=("n_common_spots", "first"),
        within_edges=("within_edges","first"), boundary_edges=("boundary_edges","first"),
        split_rate=("split_rate", "mean"), merge_rate=("merge_rate", "mean"), ari=("ari","mean"))


def plot_dose(dose, output):
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.9), sharex=True, sharey="row")
    fig.subplots_adjust(left=.10, right=.985, bottom=.15, top=.82, wspace=.16, hspace=.14)
    for col, ds in enumerate(COHORTS):
        subset = dose[dose.dataset.eq(ds)]
        label = {"DLPFC":"DLPFC  ·  12 sections, 3 donors", "prostate":"Prostate  ·  7 sections, 1 patient",
                 "breast":"Breast  ·  20 sections"}[ds]
        for row, metric in enumerate(["split_rate","merge_rate"]):
            ax=axes[row, col]
            p = subset.pivot(index="slide", columns="alpha", values=metric).sort_index(axis=1)
            for _, values in p.iterrows():
                ax.plot(p.columns, 100 * values.to_numpy(), color=COLOURS[ds], lw=.7, alpha=.23)
            m = subset.groupby("alpha")[metric].mean()
            if ds=="DLPFC":
                m = subset.groupby(["donor","alpha"])[metric].mean().groupby("alpha").mean()
            ax.plot(m.index,100*m.to_numpy(),marker="o",ms=4.5,lw=2.25,color=COLOURS[ds])
            ax.set_xlim(-.045,.745)
            ax.set_xticks([0,.3,.7],["0","0.3","0.7"])
            ax.grid(axis="y",color="#e1e4e6",lw=.6)
            ax.spines[["right","top"]].set_visible(False)
            if row==0:ax.set_title(label,fontsize=10,pad=9)
            if row==1:ax.set_xlabel("Neighbour weight (α)",fontsize=9)
            if col==0:ax.set_ylabel("Within-region edges\nsplit (%)" if row==0 else "Boundary edges\nmerged (%)",fontsize=10)
    axes[0,0].set_ylim(bottom=0)
    axes[1,0].set_ylim(0,102)
    fig.suptitle("Smoothing changes two different spatial errors",x=.10,ha="left",y=.96,fontsize=14)
    fig.text(.10,.885,"Thin lines show sections. Bold lines give means, with equal donor weight in DLPFC.",fontsize=9)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main(folder):
    archived=pd.read_csv(folder/"archive_edges.csv")
    doses=pd.read_csv(folder/"dose_edges.csv")
    for d,key,method in [(archived,"method","archive"),(doses,"alpha","dose")]:
        if d.duplicated(["slide",key,"seed"]).any():
            raise ValueError("Duplicate section, condition and seed")
        s=section_summary(d,key)
        s.to_csv(folder/f"{method}_sections.csv",index=False)
        donors=s[s.donor.notna()].groupby(["dataset","donor",key],as_index=False).agg(
            n_sections=("slide","nunique"),split_rate=("split_rate","mean"),
            merge_rate=("merge_rate","mean"),ari=("ari","mean"))
        donors.to_csv(folder/f"{method}_donors.csv",index=False)
    s=section_summary(doses,"alpha")
    breast=s[s.dataset.eq("breast")].set_index(["slide","alpha"])
    comp=breast.xs(.3,level="alpha")[["within_edges","boundary_edges"]].copy()
    archived_mean=doses.groupby(["slide","alpha"]).archived_ari.mean().unstack()
    for metric in METRICS:
        w=breast[metric].unstack()
        comp[metric+"_alpha0"]=w[0.]
        comp[metric+"_alpha03"]=w[.3]
        comp["delta_"+metric]=w[.3]-w[0.]
    comp["archived_delta_ari"]=archived_mean[.3]-archived_mean[0.]
    comp.sort_values("archived_delta_ari").to_csv(folder/"breast_weak_vs_none.csv")
    plot_dose(s,folder/"boundary_errors.png")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("folder",type=Path)
    main(ap.parse_args().folder)
