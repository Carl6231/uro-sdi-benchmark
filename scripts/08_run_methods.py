"""08_run_methods.py
Run spatial-domain identification methods on slides.
Writes one CSV of cluster labels per (slide, method, k, seed) plus a JSON with runtime / status.
Each method follows its authors' tutorial settings; embedding methods are clustered with the same
tied-covariance Gaussian mixture (sklearn analogue of mclust 'EEE') so that K is identical across methods.

Usage: python 08_run_methods.py --slides PAT1,PAT2 --methods leiden,gmm,... --seeds 0 [--k 7] [--force]
"""
import os, sys, json, time, argparse, random, warnings, tempfile, traceback, inspect
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scanpy as sc, scipy.sparse as sp, torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench")), "envs/src/STAGATE_pyG"))
sys.path.insert(0, os.path.join(os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench")), "envs/src/SEDR"))
import bench_utils as bu

sc.settings.verbosity = 0
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def call(fn, *args, **kw):
    """Call fn with only the keyword arguments it accepts (method APIs differ slightly between versions)."""
    ok = inspect.signature(fn).parameters
    return fn(*args, **{k: v for k, v in kw.items() if k in ok})


# ---------------- non-spatial baselines ----------------
def m_leiden(a, k, seed):
    b = bu.standard_pca(a, seed=seed)
    lab, res = bu.leiden_k(b, k, seed=seed)
    return lab, {"resolution": res}


def m_gmm(a, k, seed):
    b = bu.standard_pca(a, seed=seed)
    return bu.gmm_tied(b.obsm["X_pca"], k, seed), {}


# ---------------- simple spatial baseline: neighbour-averaged PCs ----------------
def m_smooth(a, k, seed):
    b = bu.standard_pca(a, seed=seed)
    Z = b.obsm["X_pca"]
    nb, d = bu.lattice_neighbors(a.obsm["spatial"])
    Zs = np.stack([0.5 * Z[i] + 0.5 * Z[n].mean(0) if len(n) else Z[i] for i, n in enumerate(nb)])
    return bu.gmm_tied(Zs, k, seed), {}


# ---------------- SpaGCN (tutorial settings, no histology) ----------------
def m_spagcn(a, k, seed):
    import SpaGCN as spg
    b = a.copy()
    spg.prefilter_genes(b, min_cells=3)
    spg.prefilter_specialgenes(b)
    # spots left with zero counts after SpaGCN's gene filter would be silently dropped by normalize_per_cell;
    # run SpaGCN without them and give each the label of its nearest retained spot afterwards
    keep = np.asarray(b.X.sum(1)).ravel() > 0
    xy_all = a.obsm["spatial"]
    b = b[keep].copy()
    x = b.obsm["spatial"][:, 0].tolist(); y = b.obsm["spatial"][:, 1].tolist()
    adj = spg.calculate_adj_matrix(x=x, y=y, histology=False)
    try:
        sc.pp.normalize_per_cell(b)
    except Exception:
        sc.pp.normalize_total(b)
    sc.pp.log1p(b)
    if sp.issparse(b.X):                       # SpaGCN calls .A on sparse input, which recent SciPy removed
        b.X = np.asarray(b.X.todense())
    l = spg.search_l(0.5, adj, start=0.01, end=1000, tol=0.01, max_run=100)
    res = spg.search_res(b, adj, l, k, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20,
                         r_seed=seed, t_seed=seed, n_seed=seed)
    clf = spg.SpaGCN(); clf.set_l(l)
    set_seed(seed)
    clf.train(b, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, _ = clf.predict()
    refined = spg.refine(sample_id=b.obs.index.tolist(), pred=[int(v) for v in y_pred], dis=adj, shape="hexagon")
    y_pred = np.asarray(y_pred).astype(int); refined = np.asarray(refined).astype(int)
    full = np.empty(a.n_obs, dtype=int); full_r = np.empty(a.n_obs, dtype=int)
    full[keep] = y_pred; full_r[keep] = refined
    if (~keep).any():
        from scipy.spatial import cKDTree
        _, j = cKDTree(xy_all[keep]).query(xy_all[~keep])
        full[~keep] = y_pred[j]; full_r[~keep] = refined[j]
    return full, {"l": float(l), "res": float(res), "n_dropped": int((~keep).sum()), "_refined": [int(v) for v in full_r]}


# ---------------- STAGATE (PyG implementation) ----------------
def m_stagate(a, k, seed):
    import STAGATE_pyG
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    d = bu.nn_dist(b.obsm["spatial"])
    call(STAGATE_pyG.Cal_Spatial_Net, b, rad_cutoff=1.1 * d, verbose=False)
    b = call(STAGATE_pyG.train_STAGATE, b, device=DEV, random_seed=seed, verbose=False)
    return bu.gmm_tied(b.obsm["STAGATE"], k, seed), {"rad_cutoff": 1.1 * d}


# ---------------- GraphST ----------------
def m_graphst(a, k, seed):
    from GraphST import GraphST as G
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    model = G.GraphST(b, device=DEV, random_seed=seed)
    b = model.train()
    emb = PCA(n_components=20, random_state=42).fit_transform(np.asarray(b.obsm["emb"]).copy())
    return bu.gmm_tied(emb, k, seed), {}


# ---------------- SEDR ----------------
def m_sedr(a, k, seed):
    import SEDR
    SEDR.fix_seed(seed)
    b = a.copy()
    b.layers["count"] = b.X.toarray()
    sc.pp.filter_genes(b, min_cells=50); sc.pp.filter_genes(b, min_counts=10)
    sc.pp.normalize_total(b, target_sum=1e6)
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", layer="count", n_top_genes=2000)
    b = b[:, b.var["highly_variable"]].copy()
    sc.pp.scale(b)
    b.obsm["X_pca"] = PCA(n_components=200, random_state=42).fit_transform(np.asarray(b.X))
    g = SEDR.graph_construction(b, 12)
    net = SEDR.Sedr(b.obsm["X_pca"], g, mode="clustering", device=DEV)
    net.train_with_dec(N=1)
    feat = net.process()[0]
    return bu.gmm_tied(feat, k, seed), {}


# ---------------- SpaceFlow ----------------
def m_spaceflow(a, k, seed):
    import networkx as nx
    if not hasattr(nx, "to_scipy_sparse_matrix"):      # removed in networkx 3; SpaceFlow 1.0.4 still calls it
        nx.to_scipy_sparse_matrix = lambda G, **kw: sp.csr_matrix(nx.to_scipy_sparse_array(G, **kw))
    from SpaceFlow import SpaceFlow as SF
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sf = SF.SpaceFlow(adata=b)
    sf.preprocessing_data(n_top_genes=3000)
    tmp = tempfile.mkdtemp()
    sf.train(embedding_save_filepath=os.path.join(tmp, "emb.tsv"), spatial_regularization_strength=0.1, z_dim=50,
             lr=1e-3, epochs=1000, max_patience=50, min_stop=100, random_seed=seed, gpu=0,
             regularization_acceleration=True, edge_subset_sz=1000000)
    return bu.gmm_tied(np.asarray(sf.embedding), k, seed), {}


METHODS = {"leiden": m_leiden, "gmm": m_gmm, "smooth": m_smooth, "spagcn": m_spagcn,
           "stagate": m_stagate, "graphst": m_graphst, "sedr": m_sedr, "spaceflow": m_spaceflow}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--kdelta", default="0")      # K sensitivity: offsets added to the reference K (write --kdelta=-2,-1,1,2)
    ap.add_argument("--runs", default=bu.RUNS)    # output folder; sensitivity runs are kept apart from the main runs
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    deltas = [int(x) for x in args.kdelta.split(",")]
    runs_dir = os.path.join(bu.BASE, args.runs)
    pats = args.slides.split(",")
    slides = [s for s in bu.list_slides() if any(p in s for p in pats)]
    methods = args.methods.split(","); seeds = [int(s) for s in args.seeds.split(",")]
    print(f"device={DEV} slides={len(slides)} methods={methods} seeds={seeds}", flush=True)
    for sid in slides:
        a = bu.load_slide(sid)
        k0 = args.k or bu.choose_k(a)
        if not k0:
            print("skip (no K):", sid, flush=True); continue
        lab = bu.scored_labels(a)
        od = os.path.join(runs_dir, sid); os.makedirs(od, exist_ok=True)
        for m, k in [(m, k0 + dk) for dk in deltas for m in methods if k0 + dk >= 2]:
            for seed in seeds:
                stem = os.path.join(od, f"{m}_k{k}_s{seed}")
                if os.path.exists(stem + ".json") and not args.force:
                    try:
                        if json.load(open(stem + ".json")).get("status") == "ok":
                            continue          # finished runs are skipped; failed ones are retried
                    except Exception:
                        pass
                if DEV.type == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                t0 = time.time(); status, err, info, pred = "ok", "", {}, None
                try:
                    set_seed(seed)
                    pred, info = METHODS[m](a, k, seed)
                    pred = np.asarray(pred).astype(int)
                    assert len(pred) == a.n_obs
                except Exception as e:
                    status, err = "fail", traceback.format_exc()[-1500:]
                sec = time.time() - t0
                rec = dict(slide=sid, method=m, k=k, seed=seed, status=status, sec=round(sec, 1), n_spots=int(a.n_obs),
                           gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEV.type == "cuda" else 0.0, error=err)
                ari = float("nan")
                if status == "ok":
                    df = pd.DataFrame({"spot": a.obs_names, "cluster": pred})
                    if "_refined" in info:
                        df["cluster_refined"] = info.pop("_refined")
                    df.to_csv(stem + ".csv", index=False)
                    rec["n_found"] = int(len(np.unique(pred)))
                    if lab is not None:
                        ok = lab.notna().values
                        ari = adjusted_rand_score(lab[ok].astype(str).values, pred[ok])
                        rec["ari"] = round(float(ari), 4)
                rec["info"] = {kk: (float(v) if isinstance(v, (int, float, np.floating)) else str(v)) for kk, v in info.items()}
                json.dump(rec, open(stem + ".json", "w"))
                print(f"{sid[-22:]:>22} {m:>10} k={k} found={rec.get('n_found','-')} ARI={ari:6.3f} {sec:6.1f}s {status} {err.strip().splitlines()[-1][:150] if err else ''}", flush=True)
    print("RUN_DONE", flush=True)
