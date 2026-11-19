"""59_check_missing.py - which benchmark runs are missing, and why (feeds the 'failed runs' statement in Fig. 1a)."""
import os, glob
import pandas as pd

BASE = os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench"))
s = pd.read_csv(os.path.join(BASE, "results/scores.csv"))
print("columns:", s.columns.tolist())
key = ["slide", "k", "seed"]
for ds in ["DLPFC", "prostate", "ccRCC", "other"]:
    c = s[s["dataset"] == ds]
    ref = set(map(tuple, c[c["method"] == "gmm"][key].values))
    for m in sorted(c["method"].unique()):
        have = set(map(tuple, c[c["method"] == m][key].values))
        miss = sorted(ref - have)
        if miss:
            print(ds, m, "missing", len(miss), miss[:6])
if "error" in s.columns:
    e = s["error"].fillna("").astype(str)
    print("rows with error text:", int((e.str.len() > 0).sum()))
    print(s.loc[e.str.len() > 0, ["method", "slide", "k", "seed", "error"]].head(10).to_string())
for m in ["sedr", "spaceflow"]:
    for f in sorted(glob.glob(os.path.join(BASE, "logs", f"*{m}*")))[:20]:
        txt = open(f, errors="ignore").read()
        if "Traceback" in txt or "Error" in txt:
            print("log with error:", f)
            print(txt[-800:])
