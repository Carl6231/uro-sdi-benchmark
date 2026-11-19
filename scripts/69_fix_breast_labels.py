"""69_fix_breast_labels.py - pathologist labels of the external breast-cancer sections, made comparable to the main data.
- GSE213688 (plastic / non-plastic breast cancers): 'Artifacts' / 'Artefacts' and uncertain calls ('... ?') are not
  tissue classes and are left unscored, like 'Exclude' in the prostate data.
- 10x Block A Section 1: the published annotation has 20 numbered regions (fine_annot_type); the tissue class of each
  region (annot_type: IDC, DCIS/LCIS, Healthy, Tumor edge) is used instead, so that classes, not regions, are scored.
Overwrites obs['label'] in data/processed/<slide>.h5ad and writes results/tables/breast_sections.csv.
"""
import os, sys, glob
import numpy as np, pandas as pd, anndata as ad
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

ANN = os.path.join(bu.BASE, "meta/stimage/annotation")
rows = []
for f in sorted(glob.glob(os.path.join(bu.PROC, "GSE213688_*.h5ad")) + glob.glob(os.path.join(bu.PROC, "Human_Breast_10X_06*.h5ad"))):
    sid = os.path.basename(f)[:-5]
    a = ad.read_h5ad(f)
    if "label" not in a.obs:
        rows.append(dict(slide=sid, n_spots=a.n_obs, k=None, classes="(no annotation)")); continue
    lab = a.obs["label"].astype(object)
    if "Block_A_Section_1" in sid:
        an = pd.read_csv(os.path.join(ANN, sid + "_anno.csv"))
        an = an.set_index(an.columns[0])
        lab = an["annot_type"].reindex(a.obs_names).astype(object)
    s = lab.astype(str)
    bad = s.isin(["nan", "None", ""]) | s.str.lower().str.startswith("artifact") | s.str.lower().str.startswith("artefact") | s.str.contains(r"\?")
    lab = lab.where(~bad.values)
    a.obs["label"] = pd.Categorical(lab)
    a.write_h5ad(f + ".tmp", compression="gzip"); os.replace(f + ".tmp", f)
    vc = lab.dropna().value_counts()
    rows.append(dict(slide=sid, n_spots=a.n_obs, k=int((vc >= bu.MIN_CLASS_SPOTS).sum()),
                     classes="; ".join(f"{k}:{v}" for k, v in vc.items())))
    print(sid, rows[-1]["k"], rows[-1]["classes"], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(bu.BASE, "results/tables/breast_sections.csv"), index=False)
print("BREAST_LABELS_DONE")
