"""62_make_sim.py - simulated sections whose true domains have a controlled spatial scale.

Expression is real: every simulated spot receives the raw-count profile of a real spot drawn from the matching
annotated class of a source dataset (no synthetic counts). Only the layout of the domains is simulated:
K independent Gaussian random fields with a squared-exponential correlation length ell (in spot spacings) are drawn
on a Visium-like hexagonal grid and each spot takes the domain with the largest field value. Small ell gives
fragmented, salt-and-pepper domains; large ell gives large compact domains.

Sources
- prostate: Erickson patient 1, 3 coarse classes (tumour = GG1/GG2/GG4/GG4 cribriform, benign = Benign/Benign*, stroma)
- dlpfc: Maynard DLPFC, 5 layers (L1, L3, L5, L6, WM)
Grid: 50 rows x 60 spots = 3,000 spots, centre-to-centre distance 100.
Scales ell: 0.6, 1, 1.6, 2.5, 4, 6.5 spot spacings; 3 replicates per scale and source (36 sections).
Writes data/processed/SIM_<source>_l<10*ell>_r<rep>.h5ad (obs: label, src_spot; obsm: spatial).
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

SOURCES = {
    "prostate": ("Erickson", {"GG1": "Tumour", "GG2": "Tumour", "GG4": "Tumour", "GG4 Cribriform": "Tumour",
                              "Benign": "Benign", "Benign*": "Benign", "Stroma": "Stroma"}),
    "dlpfc": ("Maynard", {"L1": "L1", "L3": "L3", "L5": "L5", "L6": "L6", "WM": "WM"}),
}
SCALES = [0.6, 1.0, 1.6, 2.5, 4.0, 6.5]
REPS = [1, 2, 3]
NR, NC, SPACING = 50, 60, 100.0


def hex_grid():
    xy = [((2 * j + r % 2) * SPACING / 2, r * SPACING * np.sqrt(3) / 2) for r in range(NR) for j in range(NC)]
    return np.array(xy, dtype=float)


def grf(xy, ell, rng, n_feat=2000):
    """Random-Fourier-feature draw of a Gaussian process with squared-exponential kernel, length scale ell spacings."""
    w = rng.normal(0.0, 1.0 / (ell * SPACING), size=(n_feat, 2))
    b = rng.uniform(0, 2 * np.pi, n_feat)
    return np.sqrt(2.0 / n_feat) * np.cos(xy @ w.T + b).sum(1)


def load_pool(pattern, cmap):
    """Raw counts of all annotated spots of the source sections, restricted to genes shared by every section."""
    slides = bu.list_slides(pattern)
    ads = []
    for sid in slides:
        a = bu.load_slide(sid)
        lab = bu.scored_labels(a)
        cls = lab.map(cmap)
        keep = cls.notna().values
        b = a[keep].copy()
        b.obs = pd.DataFrame({"cls": cls[keep].values, "src_spot": [f"{sid}|{s}" for s in b.obs_names]}, index=b.obs_names)
        b.obs_names = b.obs["src_spot"].values
        ads.append(b)
    genes = sorted(set.intersection(*[set(b.var_names) for b in ads]))
    pool = ad.concat([b[:, genes] for b in ads], join="inner")
    pool.X = sp.csr_matrix(pool.X)
    return pool


if __name__ == "__main__":
    xy = hex_grid()
    summary = []
    for si, (src, (pattern, cmap)) in enumerate(SOURCES.items()):
        pool = load_pool(pattern, cmap)
        classes = sorted(pool.obs["cls"].unique())
        idx_by = {c: np.where(pool.obs["cls"].values == c)[0] for c in classes}
        print(src, "pool:", pool.n_obs, "spots,", pool.n_vars, "genes;", {c: len(v) for c, v in idx_by.items()}, flush=True)
        for ei, ell in enumerate(SCALES):
            for rep in REPS:
                rng = np.random.default_rng(10000 * si + 100 * ei + rep)
                F = np.stack([grf(xy, ell, rng) for _ in classes], axis=1)
                lab = np.array(classes)[F.argmax(1)]
                pick = np.empty(len(xy), dtype=int)
                for c in classes:
                    where = np.where(lab == c)[0]
                    src_idx = idx_by[c]
                    pick[where] = rng.choice(src_idx, size=len(where), replace=len(where) > len(src_idx))
                X = pool.X[pick].astype(np.float32)
                sid = f"SIM_{src}_l{int(round(10 * ell)):02d}_r{rep}"
                obs = pd.DataFrame({"label": lab, "src_spot": pool.obs_names[pick]}, index=[f"s{i:04d}" for i in range(len(xy))])
                out = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=pd.DataFrame(index=pool.var_names))
                out.obsm["spatial"] = xy.copy()
                out.uns["sim"] = {"source": src, "ell_spacings": ell, "replicate": rep}
                out.write_h5ad(os.path.join(bu.PROC, sid + ".h5ad"), compression="gzip")
                vc = pd.Series(lab).value_counts()
                summary.append(dict(slide=sid, source=src, ell=ell, rep=rep, k=len(classes), n_spots=len(xy),
                                    min_class=int(vc.min()), max_class=int(vc.max())))
                print(sid, dict(vc), flush=True)
    os.makedirs(os.path.join(bu.BASE, "results/tables"), exist_ok=True)
    pd.DataFrame(summary).to_csv(os.path.join(bu.BASE, "results/tables/sim_sections.csv"), index=False)
    print("SIM_DONE", flush=True)
