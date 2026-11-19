"""01_inventory_stimage.py
Inventory of STimage-1K4M: which slides carry pathologist annotation, and which are urologic.
Downloads only the small metadata + annotation CSVs (no images, no expression).
"""
import os, sys, json, collections
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import pandas as pd
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "jiawennnn/STimage-1K4M"
OUT = os.path.join(os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench")), "meta/stimage")
os.makedirs(OUT, exist_ok=True)

files = list_repo_files(REPO, repo_type="dataset")
print("total files in repo:", len(files))
top = collections.Counter(f.split("/")[0] for f in files)
print("top-level:", dict(top))
anno = sorted(f for f in files if f.startswith("annotation/"))
meta = sorted(f for f in files if f.startswith("meta/"))
print("n annotation files:", len(anno))
print("meta files:", meta)
with open(os.path.join(OUT, "repo_files.json"), "w") as fh:
    json.dump(files, fh)

def get(f):
    return hf_hub_download(REPO, f, repo_type="dataset", local_dir=OUT)

meta_paths = {}
for f in meta:
    if f.endswith(".csv") or f.endswith(".txt"):
        try:
            meta_paths[f] = get(f)
        except Exception as e:
            print("meta download failed", f, repr(e)[:200])

# newest meta csv
cands = [p for f, p in meta_paths.items() if f.endswith(".csv")]
print("meta csv candidates:", cands)
mpath = sorted(cands)[-1]
m = pd.read_csv(mpath, low_memory=False)
print("\nMETA shape:", m.shape)
print("META columns:", list(m.columns))
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
print(m.head(3).to_string())
for c in m.columns:
    if m[c].nunique() <= 60:
        print(f"\n== value counts: {c} ==")
        print(m[c].value_counts(dropna=False).to_string())

# annotation prefixes
pref = collections.Counter()
for f in anno:
    b = os.path.basename(f).replace("_anno.csv", "")
    pref[b] += 1
with open(os.path.join(OUT, "annotation_slides.txt"), "w") as fh:
    fh.write("\n".join(sorted(pref)))
print("\nannotation slide ids (first 200):")
for b in sorted(pref)[:200]:
    print(" ", b)
