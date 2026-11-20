"""92_additional_file2.py - Additional file 2 (PDF): Figure S1, TLS calls in the three Zurich ccRCC sections.

Wraps external_validation/zurich_tls_maps.png (made by external_validation/zurich_tls_pilot.py) into one A4 page
with its legend. Usage: python 92_additional_file2.py <png> <out.pdf>
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

png, out = sys.argv[1], sys.argv[2]
img = mpimg.imread(png)
fig = plt.figure(figsize=(8.27, 11.69))
ax = fig.add_axes([0.06, 0.42, 0.88, 0.88 * img.shape[0] / img.shape[1] * 8.27 / 11.69])
ax.imshow(img); ax.axis("off")
fig.text(0.06, 0.95, "Additional file 2", fontsize=12, fontweight="bold")
fig.text(0.06, 0.38, (
    "Figure S1. TLS calls in three independent ccRCC sections (Zurich, KC1-KC3; Zenodo 14620362).\n"
    "Columns: H&E image; reference TLS annotation; the cluster with the highest TLS programme score without\n"
    "smoothing and after PCA-smooth with alpha = 0.3 and 0.7 (K = 6, seed 0). PCs were computed without the TLS\n"
    "marker genes. Green: reference TLS spots called TLS; blue: reference TLS spots missed; orange: annotated\n"
    "non-TLS spots called TLS; grey: annotated non-TLS spots; light grey: unassigned (not scored).\n"
    "F1 is given for each map. Scores of all 27 runs, hits per reference TLS patch and a control with shuffled\n"
    "neighbours are in the code repository (external_validation)."), fontsize=8.2, va="top", linespacing=1.5)
fig.savefig(out)
print("saved", out)
