"""43_label_free_diagnostic.py
Can a user tell, without any annotation, whether spatial smoothing is likely to help on a section?
Candidate label-free statistic: PAS (patchiness) of the non-spatial GMM clustering of that section.
Compares it per section with (a) the patchiness of the reference labels and (b) the gain of the 9 spatial methods over
GMM (ARI). Output: results/tables/label_free_diagnostic.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

T = os.path.join(bu.BASE, "results/tables")
S = pd.read_csv(os.path.join(bu.BASE, "results/scores.csv")); S = S[S["status"] == "ok"]
SP9 = ["graphst", "stagate", "sedr", "spagcn", "spaceflow", "bayesspace", "banksy", "drsc", "smooth", "bass", "precast"]   # all spatial methods
G = pd.read_csv(os.path.join(T, "label_patches_slide.csv")).set_index("slide")
d = S[S["dataset"].isin(["DLPFC", "prostate"])]
ari = d.groupby(["slide", "method"])["ari"].mean().unstack()
SP9 = [m for m in SP9 if m in ari]
gm = d[d["method"] == "gmm"].groupby("slide")[["pas", "chaos"]].mean().add_prefix("gmm_")
D = pd.DataFrame({"dataset": d.groupby("slide")["dataset"].first(), "gain9": ari[SP9].mean(1) - ari["gmm"],
                  "best_spatial_minus_gmm": ari[SP9].max(1) - ari["gmm"]}).join(gm).join(G[["boundary_frac"]])
D.to_csv(os.path.join(T, "label_free_diagnostic.csv"))
pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
print(D.sort_values("gmm_pas").to_string())
for scope in ["all", "DLPFC", "prostate"]:
    x = D if scope == "all" else D[D["dataset"] == scope]
    r1 = spearmanr(x["gmm_pas"], x["boundary_frac"]); r2 = spearmanr(x["gmm_pas"], x["gain9"])
    r3 = spearmanr(x["boundary_frac"], x["gain9"])
    print(f"{scope:>8} n={len(x):2d} | GMM-PAS vs label boundary: rho={r1[0]:.2f} P={r1[1]:.3g} | GMM-PAS vs gain: "
          f"rho={r2[0]:.2f} P={r2[1]:.3g} | label boundary vs gain: rho={r3[0]:.2f} P={r3[1]:.3g}")
