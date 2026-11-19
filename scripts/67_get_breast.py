"""67_get_breast.py - external tumour data: download the pathologist-annotated breast-cancer Visium sections of
STimage-1K4M (GSE213688, 14 sections; 10x Genomics breast cancer, 2 sections): expression + coordinates, no images.
The annotation tables are already in meta/stimage/annotation. Safe to re-run.
"""
import os, json, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

REPO = "jiawennnn/STimage-1K4M"
BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
RAW = os.path.join(BASE, "data/raw/stimage")
files = json.load(open(os.path.join(BASE, "meta/stimage/repo_files.json")))
want = [f for f in files if f.startswith("Visium/") and ("/gene_exp/" in f or "/coord/" in f)
        and (os.path.basename(f).startswith("GSE213688_") or os.path.basename(f).startswith("Human_Breast_10X_"))]
print("files:", len(want), flush=True)
ok = fail = 0
for f in want:
    for attempt in range(4):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW); ok += 1
            print(time.strftime("%H:%M:%S"), "ok", f, flush=True)
            break
        except Exception as e:
            if attempt == 3:
                fail += 1; print("FAILED", f, repr(e)[:160], flush=True)
            time.sleep(5 * (attempt + 1))
print(f"BREAST_DL_DONE ok={ok} fail={fail}", flush=True)
