"""78_table_datasets.py - Additional file tables describing the data: one row per dataset group (TableS1_datasets.csv)
and one row per section (TableS2_sections.csv). All real sections except Wu et al. 2021 were obtained from STimage-1K4M
(Hugging Face dataset jiawennnn/STimage-1K4M, Chen et al., NeurIPS 2024); the original studies are listed per group.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

T = os.path.join(bu.BASE, "results/tables")
STIM = "STimage-1K4M (Hugging Face: jiawennnn/STimage-1K4M)"
# group: (slide prefix, tissue, reference labels, use, original study, accession of the original data, obtained from)
GROUPS = [
    ("DLPFC", "Human_Brain_Maynard", "Human dorsolateral prefrontal cortex (3 donors)", "Manual cortical layers L1-L6 and white matter",
     "Labelled benchmark (laminar reference)", "Maynard et al., Nat Neurosci 2021 (PMID 33558695)", "spatialLIBD; Globus jhpce#HumanPilot10x", STIM),
    ("Prostate cancer", "Human_Prostate_Erickson", "Radical prostatectomy, patient 1 (Visium)", "Pathologist annotation (Gleason grade groups, benign, stroma and other tissue)",
     "Labelled benchmark", "Erickson et al., Nature 2022 (PMID 35948708)", "Mendeley Data doi:10.17632/svw96g68dv", STIM),
    ("ccRCC", "GSE175540", "Clear-cell renal cell carcinoma (Visium)", "Pathologist annotation incl. tertiary lymphoid structures (TLS)",
     "Labelled benchmark (TLS detection)", "Meylan et al., Immunity 2022 (PMID 35231421)", "GEO GSE175540", STIM),
    ("Kidney (KPMP atlas)", "GSE183456", "Human kidney: reference, acute and chronic injury", "None used", "Unlabelled: label-free marker analysis, dose",
     "Lake et al., Nature 2023 (PMID 37468583)", "GEO GSE183456", STIM),
    ("Kidney papilla", "GSE206306", "Human kidney papilla (stone disease)", "None used", "Unlabelled: label-free marker analysis, dose",
     "Canela et al., Nat Commun 2023 (PMID 37468493)", "GEO GSE206306", STIM),
    ("Kidney (AKI)", "GSE226533", "Human kidney, acute kidney injury", "None used", "Unlabelled: label-free marker analysis, dose",
     "GEO series GSE226533 (resident macrophage subpopulations in AKI)", "GEO GSE226533", STIM),
    ("Kidney (10x)", "Human_Kidney_10X", "Human kidney (FFPE, 10x Genomics demonstration data)", "None used",
     "Unlabelled: label-free marker analysis, dose", "10x Genomics public dataset", "10x Genomics datasets", STIM),
    ("Prostate (10x)", "Human_Prostate_10X", "Human prostate, normal and cancer (FFPE, 10x Genomics demonstration data)", "None used",
     "Unlabelled: label-free marker analysis, dose", "10x Genomics public datasets", "10x Genomics datasets", STIM),
    ("Breast cancer (TNBC)", "GSE213688", "Triple-negative breast cancer incl. claudin-low and metaplastic", "Pathologist annotation (via STimage-1K4M)",
     "External validation", "Coutant et al., Lab Invest 2023 (PMID 37813278)", "GEO GSE213688", STIM),
    ("Breast cancer (10x)", "Human_Breast_10X_06", "Invasive ductal carcinoma (10x Genomics demonstration data)", "Annotation provided with STimage-1K4M",
     "External validation", "10x Genomics public datasets", "10x Genomics datasets", STIM),
    ("Breast cancer (Wu)", "WU2021_", "Primary breast cancers (ER+, HER2+ and TNBC)", "Pathologist spot classification",
     "External validation", "Wu et al., Nat Genet 2021 (PMID 34493872)", "Zenodo doi:10.5281/zenodo.4739739", "Zenodo 4739739"),
    ("Simulation", "SIM_", "Simulated domain layouts (Gaussian random fields) filled with real spot profiles of prostate cancer and DLPFC",
     "Known by construction", "Controlled spatial grain", "This study", "Code in repository", "Generated (62_make_sim.py)"),
]

# sections that enter the study
real = sorted(pd.read_csv(os.path.join(bu.BASE, "results/dose/real_dose.csv"))["slide"].unique())
br = pd.read_csv(os.path.join(T, "breast_sections.csv"))
breast = sorted(br["slide"]) + bu.list_slides("WU2021_")
sim = sorted(pd.read_csv(os.path.join(T, "sim_sections.csv"))["slide"])
labelled = set(pd.read_csv(os.path.join(T, "section_overview.csv"))["slide"])

rows = []
for s in real + breast + sim:
    g = next(gr for gr in GROUPS if s.startswith(gr[1]))
    a = bu.load_slide(s)
    tot = np.asarray(a.X.sum(1)).ravel()
    k = bu.choose_k(a) if (s in labelled or s in breast or s.startswith("SIM_")) else None
    lab = bu.scored_labels(a)
    vc = lab.dropna().value_counts() if lab is not None and k else pd.Series(dtype=int)
    use = g[4]
    if s in breast and not k:
        use = "Excluded (fewer than 2 classes with >= 20 spots)"
    rows.append(dict(section=s, group=g[0], spots=a.n_obs, genes=a.n_vars, median_umi=float(np.median(tot)),
                     k=k if k else "", scored_spots=int(vc.sum()) if len(vc) else "",
                     classes="; ".join(f"{c} ({n})" for c, n in vc.items()) if len(vc) else "", use=use))
    print(s, a.n_obs, k, flush=True)
S2 = pd.DataFrame(rows)
S2.to_csv(os.path.join(T, "TableS2_sections.csv"), index=False)

out = []
for name, pre, tissue, labels, use, study, acc, src in GROUPS:
    d = S2[S2["group"] == name]
    if not len(d):
        continue
    used = d[~d["use"].str.startswith("Excluded")]
    out.append({"Dataset": name, "Tissue / condition": tissue, "Sections used": f"{len(used)}" + (f" (+{len(d) - len(used)} excluded)" if len(d) > len(used) else ""),
                "Spots per section, median (range)": f"{int(used['spots'].median()):,} ({used['spots'].min():,}-{used['spots'].max():,})",
                "Median UMI per spot (median over sections)": f"{used['median_umi'].median():,.0f}",
                "Reference labels": labels, "Use in this study": use, "Original study": study, "Original data": acc, "Obtained from": src})
S1 = pd.DataFrame(out)
S1.to_csv(os.path.join(T, "TableS1_datasets.csv"), index=False)
print(S1.to_string(index=False))
print("TABLE_DATASETS_DONE")
