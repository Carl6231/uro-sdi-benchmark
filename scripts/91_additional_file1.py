"""91_additional_file1.py - Additional file 1 of the article: all supplementary tables in one Excel workbook
(results/AdditionalFile1_tables.xlsx). Run with any Python that has pandas + openpyxl (on our server: ~/mambaforge/bin/python)."""
import os, sys, re
import numpy as np, pandas as pd
from openpyxl.styles import Font, Alignment
from openpyxl.utils import get_column_letter

BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
T = os.path.join(BASE, "results/tables")
sys.path.insert(0, os.path.join(BASE, "scripts"))
import marker_programs as mp

NAME = {"graphst": "GraphST", "smooth": "PCA-smooth", "stagate": "STAGATE", "sedr": "SEDR", "drsc": "DR-SC", "bayesspace": "BayesSpace",
        "banksy": "BANKSY", "gmm": "GMM", "spagcn": "SpaGCN", "leiden": "Leiden", "spaceflow": "SpaceFlow", "bass": "BASS",
        "precast": "PRECAST", "spagcn_refined": "SpaGCN (refined)"}


def rd(f, **kw):
    p = os.path.join(T, f)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def mname(df, col="method"):
    if df is not None and col in df:
        df[col] = df[col].map(lambda m: NAME.get(m, m))
    return df


sheets = []   # (sheet name, title, DataFrame)
sheets.append(("S1_datasets", "Table S1. Data sets: tissue, sections, size, reference labels, use, original study and accession", rd("TableS1_datasets.csv")))
sheets.append(("S2_sections", "Table S2. Every section: group, spots, genes, median UMI per spot, K, scored spots, classes, use", rd("TableS2_sections.csv")))
prog = [dict(tissue=t, programme=p, genes=", ".join(g)) for t, d in [("prostate", mp.PROSTATE), ("kidney", mp.KIDNEY),
        ("ccRCC (kidney programmes plus)", {k: v for k, v in mp.CCRCC.items() if k not in mp.KIDNEY}), ("DLPFC", mp.DLPFC),
        ("breast", mp.BREAST)] for p, g in d.items()]
sheets.append(("S3_marker_programmes", "Table S3. Marker-gene programmes used for the label-free check", pd.DataFrame(prog)))
sw = rd("software_versions.csv")
sheets.append(("S4_software", "Table S4. Software versions and hardware", sw))

S = pd.read_csv(os.path.join(BASE, "results/scores.csv")); S = S[S["status"] == "ok"]
rows = []
for ds, g in S[S["dataset"].isin(["DLPFC", "prostate", "ccRCC"])].groupby("dataset"):
    per = g.groupby(["slide", "method"])[["ari", "nmi", "ami", "ari_coarse", "tumor_f1", "tls_f1", "pas", "chaos", "sec"]].mean()
    m = per.groupby("method").mean()
    m.insert(0, "sections", per.reset_index().groupby("method")["slide"].nunique())
    m.insert(0, "dataset", ds)
    rows.append(m.reset_index())
acc = mname(pd.concat(rows)).rename(columns={"sec": "runtime_s"}).dropna(axis=1, how="all")
sheets.append(("S5_accuracy", "Table S5. Mean scores per data set and method (mean over seeds within a section, then over sections)", acc.round(4)))
vs = []
for ds in ("DLPFC", "prostate"):
    v = rd(f"vs_gmm_{ds}.csv")
    if v is not None:
        vs.append(v.assign(dataset=ds))
if vs:
    sheets.append(("S6_vs_GMM", "Table S6. Each method vs GMM (ARI difference per section; Wilcoxon signed-rank, Benjamini-Hochberg)",
                   mname(pd.concat(vs).reset_index(drop=True)).round(5)))
b = rd("bootstrap_vs_gmm.csv")
if b is not None:
    sheets.append(("S7_bootstrap", "Table S7. Bootstrap 95% CIs of the mean difference to GMM, per metric", mname(b).round(5)))
cg = rd("class_gain_vs_patchiness.csv")
if cg is not None:
    sheets.append(("S8_class_level", "Table S8. Class-level gain over GMM vs boundary share of the class (Spearman, per method)", mname(cg).round(5)))
be = rd("guide_breakeven.csv")
if be is not None:
    sheets.append(("S9_break_even", "Table S9. Break-even boundary share by class size (mixed model; 95% interval)", be.round(4)))
for f, nm, tt in [("sim_summary.csv", "S10_simulation", "Table S10. Simulation: boundary share and ARI per method by source and correlation length (mean, SD over 3 layouts)"),
                  ("sim_crossover.csv", "S11_sim_break_even", "Table S11. Simulation: break-even boundary share per method (bootstrap 95% interval)"),
                  ("ext_ari.csv", "S12_breast", "Table S12. External breast-cancer sections: ARI, NMI and PAS per section and method (mean over seeds)"),
                  ("ext_rule_test.csv", "S13_breast_rule", "Table S13. Out-of-sample test of the class-level model on breast-cancer classes"),
                  ("marker_section_delta.csv", "S14_markers", "Table S14. Label-free check: change of marker-programme eta-squared per section"),
                  ("guide_scorecard.csv", "S15_scorecard", "Table S15. Scorecard behind Figure 11a")]:
    d = rd(f)
    if d is not None:
        if nm == "S15_scorecard":
            d = d.rename(columns={d.columns[0]: "method"})
        sheets.append((nm, tt, mname(d).round(5)))

paired_tables = [
    ("S16_paired_cohorts", "paired_cohorts", "Table S16. Paired edge changes by cohort. Rates use all eligible pairs of each reference type"),
    ("S17_paired_sections", "paired_sections", "Table S17. Paired edge changes per section, averaged over three matched seeds"),
    ("S18_prostate_pairs", "prostate_label_pairs", "Table S18. Paired edge changes by original prostate pathology labels, averaged over sections containing that pair"),
]
edge_rates = ["corrected_rate", "introduced_rate", "error_rate_before", "error_rate_after", "net_error_change"]
for name, file, title in paired_tables:
    path = os.path.join(BASE, "results/boundary", file + ".csv")
    if os.path.exists(path):
        data = pd.read_csv(path)
        sheets.append((name, title, data))

out = os.path.join(BASE, "results/AdditionalFile1_tables.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as xw:
    idx = pd.DataFrame([(n, t.split(". ", 1)[1] if ". " in t else t) for n, t, d in sheets if d is not None], columns=["sheet", "content"])
    idx.to_excel(xw, sheet_name="Contents", index=False, startrow=2)
    contents = xw.sheets["Contents"]
    contents["A1"] = "Additional file 1 - supplementary tables"
    contents["A1"].font = Font(bold=True, size=12)
    contents.column_dimensions["A"].width = 24
    contents.column_dimensions["B"].width = 110
    for n, t, d in sheets:
        if d is None:
            continue
        d.to_excel(xw, sheet_name=n, index=False, startrow=2)
        ws = xw.sheets[n]
        ws["A1"] = t; ws["A1"].font = Font(bold=True)
        for c in ws[3]:
            c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
        for j, col in enumerate(d.columns, 1):
            w = max([len(str(col))] + [len(str(v)) for v in d[col].astype(str).head(200)])
            ws.column_dimensions[get_column_letter(j)].width = min(max(10, w + 2), 60)
            if col in edge_rates:
                for row in range(4, len(d) + 4):
                    ws.cell(row, j).number_format = "0.00%"
        ws.freeze_panes = "A4"
for n, t, d in sheets:
    print(f"{n:22s} {'missing' if d is None else d.shape}")
print("saved", out)
