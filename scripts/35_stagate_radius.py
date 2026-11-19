"""35_stagate_radius.py
STAGATE dose-response: radius of the spatial graph (rad_cutoff) = factor x spot spacing, everything else as in the
benchmark. factor 1.1 (6 neighbours) is the benchmark setting and is re-used from the main runs; 2.1 (18) and 3.1 (36)
are new. Output in the run format: results/runs_dose/<slide>/stagate-r<factor>_k<k>_s<seed>.csv / .json
Usage: python 35_stagate_radius.py --slides Erickson,Maynard --factors 2.1,3.1 --seeds 0,1,2
"""
import os, sys, json, time, argparse, importlib, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rm = importlib.import_module("08_run_methods")        # re-use the benchmark runner's set-up and helpers
import numpy as np, pandas as pd, scanpy as sc, torch
bu = rm.bu


def m_stagate_r(a, k, seed, factor):
    import STAGATE_pyG
    b = a.copy()
    sc.pp.filter_genes(b, min_cells=3)
    sc.pp.highly_variable_genes(b, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(b, target_sum=1e4); sc.pp.log1p(b)
    d = bu.nn_dist(b.obsm["spatial"])
    rm.call(STAGATE_pyG.Cal_Spatial_Net, b, rad_cutoff=factor * d, verbose=False)
    nnb = b.uns["Spatial_Net"].shape[0] / b.n_obs if "Spatial_Net" in b.uns else float("nan")
    b = rm.call(STAGATE_pyG.train_STAGATE, b, device=rm.DEV, random_seed=seed, verbose=False)
    return bu.gmm_tied(b.obsm["STAGATE"], k, seed), {"rad_cutoff": factor * d, "factor": factor, "mean_neighbors": nnb}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="Erickson,Maynard")
    ap.add_argument("--factors", default="2.1,3.1")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    torch.set_num_threads(2)
    pats = args.slides.split(",")
    slides = [s for s in bu.list_slides() if any(p in s for p in pats)]
    factors = [float(f) for f in args.factors.split(",")]; seeds = [int(s) for s in args.seeds.split(",")]
    runs = os.path.join(bu.BASE, "results/runs_dose")
    print(f"device={rm.DEV} slides={len(slides)} factors={factors} seeds={seeds}", flush=True)
    for sid in slides:
        a = bu.load_slide(sid); k = bu.choose_k(a)
        if not k:
            continue
        lab = bu.scored_labels(a)
        od = os.path.join(runs, sid); os.makedirs(od, exist_ok=True)
        for f in factors:
            tag = f"stagate-r{f:g}"
            for seed in seeds:
                stem = os.path.join(od, f"{tag}_k{k}_s{seed}")
                if os.path.exists(stem + ".json"):
                    try:
                        if json.load(open(stem + ".json")).get("status") == "ok":
                            continue
                    except Exception:
                        pass
                if rm.DEV.type == "cuda":
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                t0 = time.time(); status, err, info, pred = "ok", "", {}, None
                try:
                    rm.set_seed(seed)
                    pred, info = m_stagate_r(a, k, seed, f)
                    pred = np.asarray(pred).astype(int)
                    assert len(pred) == a.n_obs
                except Exception:
                    status, err = "fail", traceback.format_exc()[-1500:]
                sec = time.time() - t0
                rec = dict(slide=sid, method=tag, k=k, seed=seed, status=status, sec=round(sec, 1), n_spots=int(a.n_obs),
                           gpu_peak_mb=round(torch.cuda.max_memory_allocated() / 1e6, 1) if rm.DEV.type == "cuda" else 0.0, error=err)
                ari = float("nan")
                if status == "ok":
                    pd.DataFrame({"spot": a.obs_names, "cluster": pred}).to_csv(stem + ".csv", index=False)
                    rec["n_found"] = int(len(np.unique(pred)))
                    ok = lab.notna().values
                    ari = rm.adjusted_rand_score(lab[ok].astype(str).values, pred[ok]); rec["ari"] = round(float(ari), 4)
                rec["info"] = {kk: float(v) for kk, v in info.items()}
                json.dump(rec, open(stem + ".json", "w"))
                print(f"{sid[-22:]:>22} {tag:>12} k={k} s={seed} ARI={ari:6.3f} {sec:6.1f}s {status} "
                      f"{err.strip().splitlines()[-1][:150] if err else ''}", flush=True)
    print("RUN_DONE", flush=True)
