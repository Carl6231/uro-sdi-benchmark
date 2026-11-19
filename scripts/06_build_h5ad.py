"""06_build_h5ad.py  (parallel version)
Convert downloaded STimage-1K4M slides (dense count CSV + pixel coords [+ pathologist annotation]) to .h5ad.
- X: raw counts (sparse); checks that values are integers
- obsm['spatial']: pixel (x, y); obs['array_row','array_col']: hex-lattice indices reconstructed from pixels
- obs['label']: pathologist annotation when available
Re-runnable: skips slides that already have an .h5ad.
Usage: python 06_build_h5ad.py [substring-filter or ALL] [n_workers]
"""
import os, sys, glob, time
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
from scipy.spatial import cKDTree
from multiprocessing import Pool

BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
RAW = os.path.join(BASE, "data/raw/stimage/Visium")
ANN = os.path.join(BASE, "meta/stimage/annotation")
OUT = os.path.join(BASE, "data/processed"); os.makedirs(OUT, exist_ok=True)
flt = sys.argv[1] if len(sys.argv) > 1 else "ALL"
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 10
meta = pd.read_csv(os.path.join(BASE, "meta/stimage/meta/meta_all_gene02122025.csv"), low_memory=False).set_index("slide")


def hex_lattice(xy):
    """Reconstruct Visium-style (array_row, array_col) from pixel coordinates, robust to small rotation."""
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=7)
    d = np.median(dist[:, 1])
    vec = (xy[idx[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
    vec = vec[np.abs(np.linalg.norm(vec, axis=1) - d) < 0.15 * d]
    ang = np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))
    a1 = vec[np.abs(ang) < 20].mean(axis=0)          # horizontal neighbour
    a2 = vec[np.abs(ang - 60) < 20].mean(axis=0)     # neighbour near +60 degrees
    B = np.stack([a1, a2], axis=1)
    ij = np.linalg.solve(B, (xy - xy[0]).T).T
    ijr = np.rint(ij)
    resid = np.linalg.norm((B @ ijr.T).T + xy[0] - xy, axis=1) / d
    i, j = ijr[:, 0].astype(int), ijr[:, 1].astype(int)
    row = j - j.min()
    col = 2 * i + j
    col = col - col.min()
    return row, col, float(d), float((resid < 0.25).mean())


def build(cf):
    sid = os.path.basename(cf)[: -len("_count.csv")]
    out = os.path.join(OUT, sid + ".h5ad")
    crd = os.path.join(RAW, "coord", sid + "_coord.csv")
    t0 = time.time()
    try:
        df = pd.read_csv(cf, index_col=0)
        co = pd.read_csv(crd, index_col=0)
        common = df.index.intersection(co.index)
        df, co = df.loc[common], co.loc[common]
        X = sp.csr_matrix(df.values.astype(np.float32))
        is_int = bool(np.allclose(X.data, np.rint(X.data)))
        a = ad.AnnData(X=X, obs=pd.DataFrame(index=df.index.astype(str)), var=pd.DataFrame(index=df.columns.astype(str)))
        del df
        a.var_names_make_unique()
        xy = co[["xaxis", "yaxis"]].values.astype(float)
        a.obsm["spatial"] = xy
        a.obs["spot_radius_px"] = co["r"].values
        try:
            r, c, d, fit = hex_lattice(xy)
            a.obs["array_row"], a.obs["array_col"] = r, c
        except Exception as e:
            d, fit = np.nan, 0.0
        a.obs["slide"] = sid
        for k in ["tissue", "species", "tech", "involve_cancer"]:
            a.uns[k] = str(meta.loc[sid, k]) if sid in meta.index else "?"
        af = os.path.join(ANN, sid + "_anno.csv")
        n_lab, classes = 0, ""
        if os.path.exists(af):
            an = pd.read_csv(af)
            an = an.set_index(an.columns[0])
            lab = an.iloc[:, -1].reindex(a.obs_names)
            a.obs["label"] = pd.Categorical(lab)
            n_lab = int(lab.notna().sum())
            classes = "; ".join(f"{k}:{v}" for k, v in lab.value_counts().items())
        a.uns["counts_are_integer"] = is_int
        tmp = out + ".tmp"
        a.write_h5ad(tmp, compression="gzip")
        os.replace(tmp, out)
        tot = np.asarray(X.sum(1)).ravel(); ng = np.asarray((X > 0).sum(1)).ravel()
        row = dict(slide=sid, n_spots=a.n_obs, n_genes=a.n_vars, counts_integer=is_int, x_max=float(X.max()),
                   median_counts=float(np.median(tot)), median_genes=float(np.median(ng)), nn_dist_px=d,
                   lattice_fit=fit, n_in_anno_and_data=n_lab, classes=classes, sec=round(time.time() - t0, 1))
        print(f"{sid[-34:]:>34} spots={a.n_obs:5d} genes={a.n_vars:5d} int={is_int} medUMI={np.median(tot):7.0f} medGenes={np.median(ng):6.0f} lattice={fit:.3f} labeled={n_lab} ({time.time()-t0:.0f}s)", flush=True)
        return row
    except Exception as e:
        print("FAILED", sid, repr(e)[:200], flush=True)
        return None


if __name__ == "__main__":
    todo = []
    for cf in sorted(glob.glob(os.path.join(RAW, "gene_exp", "*_count.csv"))):
        sid = os.path.basename(cf)[: -len("_count.csv")]
        if flt != "ALL" and flt not in sid:
            continue
        if os.path.exists(os.path.join(OUT, sid + ".h5ad")):
            continue
        if not os.path.exists(os.path.join(RAW, "coord", sid + "_coord.csv")):
            continue
        todo.append(cf)
    print("to build:", len(todo), "workers:", NW, flush=True)
    with Pool(NW) as pool:
        rows = [r for r in pool.imap_unordered(build, todo) if r]
    if rows:
        s = pd.DataFrame(rows)
        p = os.path.join(BASE, "meta/h5ad_build_summary.csv")
        s.to_csv(p, mode="a", header=not os.path.exists(p), index=False)
    print("built:", len(rows), flush=True)
