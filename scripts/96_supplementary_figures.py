"""96_supplementary_figures.py - Additional file 2: Supplementary Figures S1-S12 in one PDF, one figure per A4 page with its legend.

The full multi-panel figures (results/figures_full, made by the 5x/8x scripts) become Figures S1-S4 and S6-S12; the Zurich TLS maps
(external_validation/zurich_tls_maps.png) become Figure S5. Usage: python 96_supplementary_figures.py <zurich_maps.png> <out.pdf>
"""
import os, sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_utils as bu

plt.rcParams["font.family"] = ["Nimbus Sans", "DejaVu Sans"]
FULL = os.path.join(bu.BASE, "results/figures_full")
zurich, out = sys.argv[1], sys.argv[2]

S = [
 ("Fig1_design_data_workload.png", "Study design, data and workload.",
  "a Workflow. b–d Reference annotation of one section from each labelled setting. e–g Spots per section, annotated classes with at "
  "least 20 spots (the K given to every method) and median UMIs per spot. h, i Class composition of the prostate and DLPFC sections. "
  "j Share of TLS spots in the 24 ccRCC sections; the 16 sections with at least 20 TLS spots were scored. k Properties of the 13 methods. "
  "l Run time per section (median and interquartile range). m Peak GPU memory of the GPU methods. n Runs per experiment and method. "
  "o Runs in total."),
 ("Fig2_accuracy.png", "Accuracy on DLPFC and prostate cancer, all metrics.",
  "a, b ARI per section (small points; mean over five seeds) and mean over sections (large points); dotted line, GMM. c Difference in ARI "
  "from GMM with 95% bootstrap intervals; filled, q < 0.05 (Wilcoxon signed-rank test, Benjamini–Hochberg). d, e ARI per section and "
  "method. f NMI. g ARI after merging the prostate labels into tumour, benign and stroma. h F1 of the best-matching domain for the "
  "merged tumour class. i Standard deviation of ARI over seeds. j Percentage of abnormal spots (PAS) of the domains; dashed lines, PAS of "
  "the reference labels. k F1 per merged prostate class. l Homogeneity against completeness. m ARI against PAS. n Mean ARI against "
  "median run time. o Rank of each method by metric (1 = best)."),
 ("Fig3_transfer_maps.png", "Rankings and domain maps of all methods on prostate cancer.",
  "a Mean ARI on DLPFC against mean ARI on prostate cancer. b Rank of each method in the two tissues. c Head-to-head comparisons on the "
  "seven prostate sections: number of sections on which the row method had a higher ARI than the column method. d Annotation of prostate "
  "section H1_2. e–o Domains of each method on H1_2 (seed closest to the method's mean), coloured by the annotated class they match "
  "best, with the ARI of the section."),
 ("Fig4_tls_ccrcc.png", "TLS detection in the Meylan ccRCC cohort, all methods.",
  "a TLS annotation of section GSM5924031. b–f Domain that best matches the TLS spots for five methods (K = 6, seed 0). g TLS F1 per "
  "section and mean over the 16 sections. h Difference in TLS F1 from GMM with 95% bootstrap intervals. i Precision and recall of the "
  "TLS domain. j TLS F1 with K = 6 and K = 10. k TLS F1 per section and method. l Mean spatial gain in TLS F1 against the boundary share "
  "of the TLS labels. m TLS F1 against the share of TLS spots. n Number of TLS patches against TLS spots. o Change in TLS F1 with the "
  "smoothing weight of PCA-smooth (one to three rings) and BANKSY (λ)."),
 (None, "TLS calls in three independent ccRCC sections (Zurich KC1–KC3).",
  "Columns: H&E image; reference TLS annotation; the cluster with the highest TLS programme score without smoothing and after "
  "PCA-smooth with α = 0.3 and 0.7 (K = 6, seed 0). PCs were computed without the TLS marker genes. Green, reference TLS spots called "
  "TLS; blue, reference TLS spots missed; orange, annotated non-TLS spots called TLS; grey, annotated non-TLS spots; light grey, "
  "unassigned (not scored). F1 is given for each map. Scores of all 27 runs, hits per reference TLS patch and the shuffled-neighbour "
  "control are in the code repository (external_validation)."),
 ("Fig5_label_granularity.png", "The grain of the reference labels, in detail.",
  "a–c Reference labels of three sections; black dots mark boundary spots. d–f Share of boundary spots, number of label patches and "
  "spot-weighted patch size per section. g Boundary share of each class (median over sections). h Class gain in F1 of the 11 spatial "
  "methods over GMM against the boundary share of the class, with the mixed-model fit. i Spearman correlation between class gain and "
  "boundary share per method; filled, P < 0.05. j–l Labels of one prostate section before and after a majority vote over one and two "
  "rings. m Mean ARI of each method re-scored on the smoothed prostate labels. n Number of spatial methods with a higher mean ARI than "
  "GMM; numbers on bars, boundary share. o Change in ARI of PCA-smooth at α = 1 against the boundary share of the section."),
 ("Fig6_dose_response.png", "Dose–response of spatial smoothing, in detail.",
  "a–c Change in ARI of PCA-smooth against the neighbour weight α with neighbourhoods of one, two and three rings (mean and 95% interval). "
  "d, e Change in ARI per section (one ring). f Best α against the boundary share of the labels. g PAS of the domains against α; dashed, "
  "PAS of the reference labels. h, i Change in ARI with BANKSY's spatial weight λ, overall and per section. j Change in ARI with the "
  "STAGATE graph radius. k, l Change in prostate tumour F1 and coarse ARI against α. m–o Prostate section V1_2: annotation, GMM domains "
  "(α = 0) and PCA-smooth domains at α = 1."),
 ("Fig7_simulation.png", "Simulation with controlled grain.",
  "a–c Simulated layouts at three correlation lengths ℓ, filled with real prostate tumour, benign and stroma spot profiles. d Boundary "
  "share of the simulated labels against ℓ; dashed lines, median boundary share of the real labels. e, f Mean ARI of each method by ℓ "
  "with prostate and DLPFC profiles. g Mean spatial gain over GMM against boundary share, simulated (open) and real (filled) sections. "
  "h Boundary share at which each method's gain over GMM ends, with 95% bootstrap intervals. i Ranks at the finest and coarsest grain. "
  "j Change in ARI of PCA-smooth with α by ℓ (prostate profiles). k Best α against boundary share, simulated and real sections. "
  "l Change in ARI at α = 1 with one to three rings. m–o Domains of GMM and STAGATE at ℓ = 1 and of STAGATE at ℓ = 6.5."),
 ("Fig8_external_breast.png", "External breast-cancer test, in detail.",
  "a–c Annotations of the sections with the lowest, median and highest boundary share. d Boundary share of the labels in the three data "
  "sets. e ARI per section and method; columns ordered by boundary share. f Mean spatial gain over GMM against boundary share. "
  "g Difference in ARI from GMM in breast cancer with 95% bootstrap intervals. h Ranks in the three data sets. i Class gain against "
  "boundary share with the predictions of the class-level model fitted on DLPFC and prostate. j Predicted against observed gain. "
  "k Change in ARI of PCA-smooth with α per breast section. l Best α against boundary share in all three data sets. m–o Section with the "
  "highest boundary share: domains of GMM and of the best and worst spatial method."),
 ("Fig9_label_free_markers.png", "Label-free check with marker-gene programmes, in detail.",
  "a–c Programme scores for glomerular podocytes and proximal tubule in a kidney section and for tumour epithelium in prostate section "
  "H1_2. d Change in η² relative to GMM for each method and programme. e Change in η² of the mean spatial method per section. f Change "
  "in η² per method in DLPFC (grey) and the other tissues. g–i Change in η² with α per programme, with programme genes held out; dashed, "
  "all genes. j Change in η² at α = 1 with and without the programme genes. k Label-free change in η² against the change in ARI at α = 1. "
  "l Change in η² at α = 1 against the expression coherence of the section. m–o Kidney section of panel a: top-decile glomerular spots "
  "and the domains at α = 0, 0.5 and 1."),
 ("Fig10_robustness.png", "Robustness.",
  "a–f Prostate cancer: ARI, coarse ARI and tumour F1 when K is varied from two below to two above the number of annotated classes and at "
  "each method's best K. g Ranks when one section is left out. h Ranks for each seed. i Spearman correlation between metrics across "
  "methods. j Bootstrap intervals of the difference from GMM for several metrics. k SpaGCN with and without its refinement step. "
  "l, m PAS of GMM domains as a label-free indicator of the spatial gain, at the reference K and at fixed K. n Run time against section "
  "size. o Methods with a higher mean ARI than GMM after majority-vote smoothing of the DLPFC labels."),
 ("Fig11_guide.png", "Scorecard and guide, in detail.",
  "a Scorecard of all methods; colour scaled within each column (darker = better). b Mean rank over the eight accuracy columns. c Class "
  "gain against boundary share with the class-level model and its 95% band for three class sizes. d Break-even boundary share against "
  "class size with its 95% interval. e Share of classes for which the model predicted the sign of the gain. f Change in ARI of PCA-smooth "
  "with α by boundary share, real sections (95% interval) and simulated sections (dashed). g Mean scaled accuracy against run time. "
  "h Decision guide."),
]

with PdfPages(out) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.08, 0.92, "Additional file 2", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.885, "Supplementary Figures S1–S12", fontsize=12)
    fig.text(0.08, 0.86, "When does spatial information help? Benchmarking spatial domain identification on Visium sections of "
             "urological tumours", fontsize=9, color="#52514e")
    y = 0.8
    for i, (_, t, _) in enumerate(S, 1):
        fig.text(0.08, y, f"Figure S{i}.  {t}", fontsize=10); y -= 0.03
    pdf.savefig(fig); plt.close(fig)
    for i, (f, t, leg) in enumerate(S, 1):
        img = mpimg.imread(zurich if f is None else os.path.join(FULL, f))
        h, w = img.shape[:2]
        fig = plt.figure(figsize=(8.27, 11.69))
        top, bottom_text = 0.97, 0.2
        aw = 0.9; ah = aw * h / w * 8.27 / 11.69
        if ah > top - bottom_text:
            ah = top - bottom_text; aw = ah * w / h * 11.69 / 8.27
        ax = fig.add_axes([(1 - aw) / 2, top - ah, aw, ah]); ax.imshow(img, interpolation="lanczos"); ax.axis("off")
        body = f"Figure S{i}. {t} " + leg
        fig.text(0.06, top - ah - 0.015, "\n".join(textwrap.wrap(body, 125)), fontsize=7.6, va="top", linespacing=1.45)
        pdf.savefig(fig, dpi=220); plt.close(fig)          # keeps the file under BMC's limit for additional files
print("saved", out)
