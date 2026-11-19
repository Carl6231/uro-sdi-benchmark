"""03b_download_dlpfc.py
Download the 12 annotated DLPFC slides (Maynard et al.) from STimage-1K4M as a calibration set:
we first check that our pipeline reproduces published method rankings on this standard dataset.
"""
import os, json, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

REPO = "jiawennnn/STimage-1K4M"
BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
RAW = os.path.join(BASE, "data/raw/stimage")
files = json.load(open(os.path.join(BASE, "meta/stimage/repo_files.json")))
want = [f for f in files if f.startswith("Visium/") and "/image/" not in f and "Human_Brain_Maynard_02082021_Visium_1516" in f
        or (f.startswith("Visium/") and "/image/" not in f and "Human_Brain_Maynard_02082021_Visium_1515" in f)]
print("files to fetch:", len(want)); sys.stdout.flush()
ok = fail = 0
for i, f in enumerate(want):
    for attempt in range(4):
        try:
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW); ok += 1; break
        except Exception as e:
            if attempt == 3:
                fail += 1; print("FAILED", f, repr(e)[:160])
            time.sleep(5 * (attempt + 1))
    print(f"[{time.strftime('%H:%M:%S')}] {i+1}/{len(want)} (ok={ok}, fail={fail})"); sys.stdout.flush()
print(f"DONE ok={ok} fail={fail}")
