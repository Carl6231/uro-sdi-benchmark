"""68_build_wu2021.py - six primary breast cancers (Wu et al. 2021, Nat Genet; Zenodo record 4739739) to .h5ad.
Raw counts from the filtered 10x matrices, spot positions from tissue_positions_list.csv (full-resolution pixels),
pathologist label per spot from the 'Classification' column of the metadata. 'Artefact', 'Uncertain' and missing
calls are left unscored. Output: data/processed/WU2021_<sample>.h5ad
"""
import os, sys, glob, gzip, tarfile
import numpy as np, pandas as pd, anndata as ad, scipy.io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

RAW = os.path.join(bu.BASE, "data/raw_ext/wu2021")


def _open(p):
    """some files in the archive end in .gz but are plain text: open by content, not by name"""
    with open(p, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    return gzip.open(p, "rb") if gz else open(p, "rb")


def read_mtx_dir(d):
    with _open(os.path.join(d, "matrix.mtx.gz")) as f:
        X = scipy.io.mmread(f).T.tocsr()                      # file: genes x spots
    with _open(os.path.join(d, "features.tsv.gz")) as f:
        feats = pd.read_csv(f, sep="\t", header=None)
    with _open(os.path.join(d, "barcodes.tsv.gz")) as f:
        bcs = pd.read_csv(f, sep="\t", header=None)[0].astype(str).values
    a = ad.AnnData(X=X, obs=pd.DataFrame(index=bcs),
                   var=pd.DataFrame({"gene_ids": feats[0].values}, index=feats[1 if feats.shape[1] > 1 else 0].astype(str).values))
    a.var_names_make_unique()
    return a


for f in ("filtered_count_matrices.tar.gz", "spatial.tar.gz", "metadata.tar.gz"):
    with tarfile.open(os.path.join(RAW, f)) as t:
        t.extractall(RAW, members=[m for m in t.getmembers() if not m.name.endswith((".jpg", ".png"))])
for d in sorted(glob.glob(os.path.join(RAW, "filtered_count_matrices", "*_filtered_count_matrix"))):
    smp = os.path.basename(d).replace("_filtered_count_matrix", "")
    a = read_mtx_dir(d)
    pos = pd.read_csv(os.path.join(RAW, "spatial", f"{smp}_spatial", "tissue_positions_list.csv"), header=None,
                      names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]).set_index("barcode")
    meta = pd.read_csv(os.path.join(RAW, "metadata", f"{smp}_metadata.csv"), index_col=0)
    a = a[a.obs_names.isin(pos.index)].copy()
    a.obsm["spatial"] = pos.loc[a.obs_names, ["pxl_col", "pxl_row"]].values.astype(float)
    a.obs["array_row"] = pos.loc[a.obs_names, "array_row"].values; a.obs["array_col"] = pos.loc[a.obs_names, "array_col"].values
    lab = meta["Classification"].reindex(a.obs_names).astype(object)
    bad = lab.isna() | lab.astype(str).str.strip().isin(["", "NA", "Artefact", "Uncertain", "nan"])
    a.obs["label"] = pd.Categorical(lab.where(~bad))
    a.obs["subtype"] = meta["subtype"].reindex(a.obs_names).values
    a.X = a.X.astype(np.float32)
    sid = f"WU2021_{smp}"
    a.obs["slide"] = sid
    a.write_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), compression="gzip")
    vc = a.obs["label"].value_counts()
    print(sid, a.n_obs, "spots", a.n_vars, "genes | K =", int((vc >= bu.MIN_CLASS_SPOTS).sum()), "|", dict(vc), flush=True)
print("WU_BUILD_DONE")
