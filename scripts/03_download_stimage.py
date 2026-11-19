"""03_download_stimage.py
Download expression + coordinates (no images) for human prostate and kidney Visium slides
from STimage-1K4M (via hf-mirror). Annotated slides first. Safe to re-run (skips finished files).
"""
import os, json, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = "jiawennnn/STimage-1K4M"
BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
META = os.path.join(BASE, "meta/stimage")
RAW = os.path.join(BASE, "data/raw/stimage")
os.makedirs(RAW, exist_ok=True)

uro = pd.read_csv(os.path.join(META, "urologic_slides_all.csv"))
uro = uro[(uro["species"] == "human") & (uro["tissue"].isin(["prostate", "kidney"])) & (uro["tech"] == "Visium")]
anno = set(open(os.path.join(META, "annotation_slides.txt")).read().split())
uro["annotated"] = uro["slide"].isin(anno)
uro = uro.sort_values(["annotated", "tissue"], ascending=[False, False])
uro.to_csv(os.path.join(BASE, "meta/selected_slides.csv"), index=False)
print("selected slides:", len(uro), "| annotated:", int(uro["annotated"].sum()))
print(uro.groupby(["tissue", "annotated"]).size().to_string())

files = json.load(open(os.path.join(META, "repo_files.json")))
want = []
for s in uro["slide"]:
    hit = [f for f in files if f.startswith("Visium/") and "/image/" not in f and os.path.basename(f).startswith(s + "_")]
    want.extend(hit)
print("files to fetch:", len(want))

# sizes
api = HfApi()
sizes = {}
try:
    for sub in ["Visium/gene_exp", "Visium/coord"]:
        for it in api.list_repo_tree(REPO, path_in_repo=sub, repo_type="dataset"):
            if getattr(it, "size", None) is not None:
                sizes[it.path] = it.size
    tot = sum(sizes.get(f, 0) for f in want)
    print(f"total size to fetch: {tot/1e9:.2f} GB")
except Exception as e:
    print("size listing failed:", repr(e)[:200])
sys.stdout.flush()

ok = fail = 0
for i, f in enumerate(want):
    for attempt in range(4):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW)
            ok += 1
            break
        except Exception as e:
            if attempt == 3:
                fail += 1
                print("FAILED", f, repr(e)[:160])
            time.sleep(5 * (attempt + 1))
    if (i + 1) % 10 == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(want)} done (ok={ok}, fail={fail})")
        sys.stdout.flush()
print(f"DONE ok={ok} fail={fail}")
