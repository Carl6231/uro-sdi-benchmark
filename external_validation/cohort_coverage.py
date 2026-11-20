"""Summarize HTAN tables S3/S12 and the original GEO spot annotations.

Inputs are the HTAN supplementary workbook from PMC13531001 and the public
GSE175540 tissue position and TLS annotation CSVs, as distributed by GEO.
Run with pandas and openpyxl from any directory.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent


def htan_sections():
    book = BASE / "inputs" / "HTAN_supp.xlsx"
    sections = pd.read_excel(book, sheet_name="Table S3").dropna(subset=["Section_ID"])
    sections = sections.loc[sections["Assay_Type"].eq("Visium")]
    spots = pd.read_excel(book, sheet_name="Table S12").dropna(subset=["sid", "Barcode", "label"])
    assert not spots.duplicated(["sid", "Barcode"]).any()
    annot = spots.merge(sections[["Case", "Site", "Section_ID", "Assay_Type"]], left_on="sid",
                        right_on="Section_ID", how="left", validate="many_to_one")
    assert annot["Site"].notna().all()
    prostate = annot.loc[annot["Site"].eq("Prostate")].copy()
    # The supplement lists numbered annotated regions. Retain their prefixes without
    # interpreting the numerical suffix as a Gleason grade.
    prostate["group"] = prostate["label"].str.replace(r"\s+\d+$", "", regex=True)
    assert set(prostate["group"]) == {"GP", "Normal"}
    rows = []
    for (case, section), group in prostate.groupby(["Case", "sid"]):
        row = {"case": case, "section": section, "labelled_spots": len(group)}
        for label in ["GP", "Normal"]:
            selected = group[group["group"].eq(label)]
            row[f"{label}_spots"] = len(selected)
            row[f"{label}_region_labels"] = selected["label"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def positive_patch_sizes(positive):
    remaining = set(positive.itertuples(index=False, name=None))
    offsets = [(0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            row, col = stack.pop()
            size += 1
            for dr, dc in offsets:
                point = (row + dr, col + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        sizes.append(size)
    return sorted(sizes)


def meylan_sections():
    folder = BASE / "inputs" / "GSE175540"
    rows = []
    for coords in sorted(folder.glob("GSM*tissue_positions_list.csv.gz")):
        acc = coords.name.split("_", 1)[0]
        pos = pd.read_csv(coords, header=None, names=["barcode", "in_tissue", "row", "col", "y", "x"])
        tissue = pos[pos.in_tissue.eq(1)].set_index("barcode")
        assert tissue.index.is_unique and not tissue.duplicated(["row", "col"]).any()
        af = list(folder.glob(f"{acc}*TLS_annotation.csv.gz"))
        if af:
            labels = pd.read_csv(af[0], index_col="Barcode").iloc[:, 0]
            assert labels.index.is_unique
            truth = labels.reindex(tissue.index)
        else:
            truth = pd.Series(index=tissue.index, dtype=object)
        positive = truth.eq("TLS")
        sizes = positive_patch_sizes(tissue.loc[positive, ["row", "col"]])
        ntls = int(positive.sum())
        nneg = int(truth.eq("NO_TLS").sum())
        assert sum(sizes) == ntls
        rows.append({"accession": acc, "annotation_file": bool(af), "tissue_spots": len(tissue),
                     "TLS": ntls, "NO_TLS": nneg, "T_agg": int(truth.eq("T_agg").sum()),
                     "unannotated": int(truth.isna().sum()), "eligible_original_TLS_F1": ntls >= 20 and nneg > 0,
                     "positive_patches": len(sizes), "patches_under_20_spots": sum(size < 20 for size in sizes),
                     "patch_sizes_spots": ",".join(map(str, sizes))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    htan_sections().to_csv(BASE / "htan_prostate_sections.csv", index=False)
    meylan_sections().to_csv(BASE / "meylan_tls_sections.csv", index=False)
