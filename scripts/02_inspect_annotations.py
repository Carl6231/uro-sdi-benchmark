"""02_inspect_annotations.py
Compact look at (a) every annotated slide: tissue + label classes, (b) all prostate / kidney / bladder slides.
"""
import os, glob, json
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "jiawennnn/STimage-1K4M"
OUT = os.path.join(os.environ.get("URO_SDI_BASE", os.path.expanduser("~/uro_sdi_bench")), "meta/stimage")
m = pd.read_csv(os.path.join(OUT, "meta/meta_all_gene02122025.csv"), low_memory=False)
m["title_short"] = m["title"].astype(str).str.slice(0, 90)
files = json.load(open(os.path.join(OUT, "repo_files.json")))
anno = sorted(f for f in files if f.startswith("annotation/"))

rows = []
for f in anno:
    p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=OUT)
    sid = os.path.basename(f).replace("_anno.csv", "")
    a = pd.read_csv(p)
    # label column = last non-id column
    cols = list(a.columns)
    labcol = cols[-1]
    vc = a[labcol].astype(str).value_counts(dropna=False)
    mm = m[m["slide"] == sid]
    rows.append(dict(slide=sid,
                     tissue=mm["tissue"].iloc[0] if len(mm) else "?",
                     tech=mm["tech"].iloc[0] if len(mm) else "?",
                     spot_num=int(mm["spot_num"].iloc[0]) if len(mm) else -1,
                     n_anno=len(a), cols="|".join(cols), n_class=vc.shape[0],
                     classes="; ".join(f"{k}:{v}" for k, v in vc.items())))
inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(OUT, "annotated_slides_inventory.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 200, "display.max_rows", 500)

print("== annotated slides by study prefix / tissue ==")
inv["study"] = inv["slide"].str.replace(r"(_GSM\d+|_Visium.*|_ST_.*)$", "", regex=True)
print(inv.groupby(["study", "tissue", "tech"]).agg(n=("slide", "size"), spots=("spot_num", "sum"),
      min_class=("n_class", "min"), max_class=("n_class", "max")).to_string())

for key in ["GSE175540", "GSE193460", "GSE213688", "Erickson"]:
    sub = inv[inv["slide"].str.contains(key)]
    print(f"\n== {key}: {len(sub)} slides; columns: {sub['cols'].iloc[0]} ==")
    for _, r in sub.iterrows():
        print(f"  {r['slide'][-22:]:>22} spots={r['spot_num']:>5} anno={r['n_anno']:>5} | {r['classes'][:230]}")

print("\n== all urologic slides in STimage-1K4M ==")
uro = m[m["tissue"].astype(str).str.contains("kidney|prostate|bladder|renal|urothel|testis", case=False)]
uro = uro.assign(study=uro["slide"].str.replace(r"(_GSM\d+)$", "", regex=True))
g = uro.groupby(["tissue", "species", "tech", "study", "involve_cancer", "title_short"]).agg(n=("slide", "size"), spots=("spot_num", "sum")).reset_index()
print(g.to_string())
uro.drop(columns=["abstract"]).to_csv(os.path.join(OUT, "urologic_slides_all.csv"), index=False)
