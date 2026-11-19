"""geom_utils.py - label geometry and class-level scores shared by the follow-up analyses (same definitions as 36_robustness.py)."""
import os
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import bench_utils as bu


def load_labels(sid):
    a = ad.read_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), backed="r")
    obs = a.obs.copy(); xy = np.asarray(a.obsm["spatial"], dtype=float); a.file.close()
    if "label" not in obs:
        return None, xy, obs.index
    lab = obs["label"].astype(object)
    lab = lab.where(~lab.astype(str).isin(bu.EXCLUDE_LABELS)).where(lab.notna())
    return lab.map(lambda v: v if pd.isna(v) else str(v)), xy, obs.index


def ring_edges(xy, factor=1.1):
    nb, _ = bu.lattice_neighbors(xy, factor)
    r = np.concatenate([np.full(len(v), i, dtype=int) for i, v in enumerate(nb)])
    return nb, r, np.concatenate(nb).astype(int)


def label_patches(v, r, c):
    """connected patches of identical scored labels (ring-1 adjacency), patch size per spot, boundary spots, scored mask"""
    s = pd.notna(v); n = len(v)
    same = s[r] & s[c] & (v[r] == v[c])
    G = sp.csr_matrix((np.ones(int(same.sum())), (r[same], c[same])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    psize = np.zeros(n)
    psize[s] = pd.Series(comp[s]).value_counts().reindex(comp[s]).values
    diff = s[r] & s[c] & (v[r] != v[c])
    bnd = np.zeros(n, dtype=bool); bnd[np.unique(r[diff])] = True
    return comp, psize, bnd, s


def geometry(sid, dataset):
    """(section row, list of class rows) with boundary share and patch statistics of the reference labels."""
    lab, xy, _ = load_labels(sid)
    if lab is None or lab.notna().sum() == 0:
        return None, []
    v = lab.values
    nb1, r1, c1 = ring_edges(xy)
    comp, psize, bnd, s = label_patches(v, r1, c1)
    vc = pd.Series(v[s]).value_counts(); classes = list(vc[vc >= bu.MIN_CLASS_SPOTS].index)
    row = dict(slide=sid, dataset=dataset, n_scored=int(s.sum()), n_classes=len(classes),
               n_patches=int(len(np.unique(comp[s]))), spot_weighted_patch=float(psize[s].mean()),
               boundary_frac=float(bnd[s].mean()))
    crow = []
    for cl in classes:
        m = s & (v == cl)
        pc = pd.Series(comp[m]).value_counts()
        crow.append(dict(slide=sid, dataset=dataset, cls=cl, n_spots=int(m.sum()), n_patches=int(len(pc)),
                         spot_weighted_patch=float(psize[m].mean()), boundary_frac=float(bnd[m].mean())))
    return row, crow


def best_f1_by_class(yt, yp, classes):
    ct = pd.crosstab(yt, yp); col = ct.sum(0).values
    return {cl: float((2 * ct.loc[cl].values / (ct.loc[cl].values.sum() + col)).max()) for cl in classes if cl in ct.index}
