"""14_export_for_r.py
Export slides for the R methods: counts (MatrixMarket, genes x spots, genes seen in >= 3 spots), gene / barcode lists,
and coordinates including hex-lattice array indices rebuilt from pixel positions (orientation-agnostic).
Usage: python 14_export_for_r.py PATTERN[,PATTERN] [n_workers]
"""
import os, sys
import numpy as np, pandas as pd, scipy.sparse as sp, scipy.io as sio
from scipy.spatial import cKDTree
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

OUT = os.path.join(bu.BASE, "data/r_input")


def hex_lattice(xy):
    """(array_row, array_col) in Visium convention from pixel coordinates; works for any lattice rotation."""
    dist, idx = cKDTree(xy).query(xy, k=7)
    d = np.median(dist[:, 1])
    vec = (xy[idx[:, 1:]] - xy[:, None, :]).reshape(-1, 2)
    vec = vec[np.abs(np.linalg.norm(vec, axis=1) - d) < 0.15 * d]
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    th0 = np.angle(np.mean(np.exp(1j * 6 * ang))) / 6            # lattice orientation modulo 60 degrees

    def pick(t):
        dd = np.angle(np.exp(1j * (ang - t)))
        return vec[np.abs(dd) < np.radians(15)].mean(axis=0)

    B = np.stack([pick(th0), pick(th0 + np.pi / 3)], axis=1)
    ijr = np.rint(np.linalg.solve(B, (xy - xy[0]).T).T)
    resid = np.linalg.norm((B @ ijr.T).T + xy[0] - xy, axis=1) / d
    i, j = ijr[:, 0].astype(int), ijr[:, 1].astype(int)
    row = j - j.min(); col = 2 * i + j; col = col - col.min()
    return row, col, float((resid < 0.25).mean())


def export(sid):
    od = os.path.join(OUT, sid)
    if os.path.exists(os.path.join(od, "DONE")):
        return sid, -1.0
    os.makedirs(od, exist_ok=True)
    a = bu.load_slide(sid)
    keep = np.asarray((a.X > 0).sum(0)).ravel() >= 3
    a = a[:, keep].copy()
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    row, col, fit = hex_lattice(xy)
    # every spot must have a unique lattice position, otherwise the fit is not trusted
    uniq = len(set(zip(row.tolist(), col.tolist()))) == a.n_obs
    sio.mmwrite(os.path.join(od, "counts.mtx"), sp.coo_matrix(a.X.T).astype(np.int32))
    open(os.path.join(od, "genes.txt"), "w").write("\n".join(a.var_names) + "\n")
    open(os.path.join(od, "barcodes.txt"), "w").write("\n".join(a.obs_names) + "\n")
    pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "array_row": row, "array_col": col}, index=a.obs_names).to_csv(os.path.join(od, "coords.csv"))
    open(os.path.join(od, "DONE"), "w").write(f"lattice_fit={fit:.4f} unique={uniq}\n")
    return sid, fit if uniq else -fit


if __name__ == "__main__":
    pats = sys.argv[1].split(",") if len(sys.argv) > 1 else [""]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    slides = [s for s in bu.list_slides() if any(p in s for p in pats)]
    print("slides:", len(slides), flush=True)
    with Pool(nw) as pool:
        for sid, fit in pool.imap_unordered(export, slides):
            print(f"{sid[-40:]:>40} lattice_fit={fit:.3f}", flush=True)
    print("EXPORT_DONE")
