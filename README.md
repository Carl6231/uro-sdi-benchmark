# When does spatial information help? A benchmark of spatial-domain identification in urological tumours

Code, result tables and figures for the study *"When does spatial information help? Benchmarking spatial domain
identification on Visium sections of urological tumours"* (manuscript in preparation).

We compare 13 clustering approaches on public 10x Visium data:

| Group | Methods |
|---|---|
| Published spatial methods (10) | GraphST, STAGATE, SEDR, SpaGCN, SpaceFlow, BayesSpace, BANKSY, DR-SC, BASS, PRECAST |
| Simple spatial baseline | PCA-smooth (PCA scores averaged with the neighbouring spots, then a Gaussian mixture) |
| Non-spatial baselines | GMM (Gaussian mixture on PCA scores), Leiden |

on

* **labelled references**: 12 DLPFC sections (cortical layers), 7 prostate-cancer sections (pathologist Gleason
  annotation, Erickson et al. 2022) and 24 clear-cell renal cell carcinoma sections (tertiary lymphoid structures,
  Meylan et al. 2022);
* **unlabelled sections**: 31 kidney and 4 prostate sections (label-free analyses);
* **external validation**: annotated breast-cancer sections (GSE213688, 10x Genomics, Wu et al. 2021);
* **simulations**: 36 sections with real expression profiles and domain layouts of controlled spatial grain.

The main question is not "which method is best" but *when* spatial smoothing helps: the gain of spatial methods over a
non-spatial Gaussian mixture shrinks as the reference domains become smaller and more fragmented, and turns into a loss
for patchy tumour structures.

## Repository layout

```
scripts/            analysis code (Python 3.10, R 4.5); numbered in the order they were written
  bench_utils.py      shared helpers (data loading, K choice, neighbours, GMM)
  geom_utils.py       geometry of reference labels (boundary share, patches)
  marker_programs.py  marker-gene programmes used for the label-free analysis
  fig_common.py       figure style and helpers
  mstyle.py           style of the six main figures (builds on fig_common.py)
environment/        conda environment, R package installation, software versions
results/tables/     per-section and per-method summaries; results/AdditionalFile1_tables.xlsx holds Supplementary Tables
                    S1-S18 and results/AdditionalFile2_SupplementaryFigures.pdf Supplementary Figures S1-S12 with legends
figures/            main Figures 1-6 (PNG and PDF); figures/supplementary/ holds Supplementary Figures S1-S12
jobs/               R job lists used for the supplementary runs and BANKSY parameter sweep
run_all.sh          the complete analysis, step by step
```

## Installation

1. Python environment (conda/mamba; a CUDA GPU is needed for GraphST, STAGATE, SEDR and SpaceFlow):

   ```bash
   conda env create -f environment/uro_sdi.yml        # or follow environment/README.md
   conda activate uro_sdi
   ```

   STAGATE (pyG version) and SEDR are used from their GitHub sources; `environment/README.md` gives the commands.

2. R packages (R 4.5): `Rscript environment/install_R_packages.R` installs BayesSpace, BANKSY, DR.SC, PRECAST,
   SPARK and BASS into a project library (`envs/Rlib`).

Exact versions are listed in `environment/software_versions.csv`.

## Data

All sections except Wu et al. 2021 are downloaded from the STimage-1K4M collection on Hugging Face
(`jiawennnn/STimage-1K4M`); Wu et al. 2021 comes from Zenodo (record 4739739). The original studies and accession
numbers of every section are listed in `results/tables/TableS1_datasets.csv` and `TableS2_sections.csv`.
No data are redistributed here.

## Reproducing the analysis

Set the project folder and run the steps in `run_all.sh` (they can be run one by one; the full benchmark takes
several days on one workstation with one GPU):

```bash
export URO_SDI_BASE=/path/to/project      # default: ~/uro_sdi_bench
bash run_all.sh
```

Main figures (`figures/`):

| Figure | Script | Content |
|---|---|---|
| 1 | `m1_fig1.py` | study design; accuracy on DLPFC and prostate cancer; rankings do not transfer; domain maps |
| 2 | `m2_fig2.py` | tertiary lymphoid structures in ccRCC: Meylan cohort and the independent Zurich sections |
| 3 | `m3_fig3.py` | the grain of the reference labels explains the spatial gain; label smoothing; simulation |
| 4 | `m4_fig4.py` | smoothing dose; neighbour-pair errors corrected and introduced by smoothing |
| 5 | `m5_fig5.py` | external test in breast cancer |
| 6 | `m6_fig6.py` | label-free check with marker-gene programmes; scorecard and decision guide |

Supplementary figures (`figures/supplementary/`, collected with legends in
[Additional file 2](results/AdditionalFile2_SupplementaryFigures.pdf) by `96_supplementary_figures.py`):

| Figure | Script | Content |
|---|---|---|
| S1 | `51_fig1_overview.py` | study design, data, methods, cost |
| S2 | `52_fig2_accuracy.py` | accuracy on DLPFC and prostate cancer, all metrics |
| S3 | `53_fig3_transfer_maps.py` | rankings across tissues; domain maps of all 13 methods |
| S4 | `54_fig4_tls.py` | tertiary lymphoid structures in the Meylan ccRCC sections |
| S5 | `external_validation/zurich_tls_pilot.py` | TLS calls in the three Zurich ccRCC sections |
| S6 | `55_fig5_mechanism.py` | label grain in detail |
| S7 | `56_fig6_dose.py` | smoothing dose-response in detail |
| S8 | `81_fig7_simulation.py` | simulation with controlled grain |
| S9 | `82_fig8_external.py` | external test in breast cancer in detail |
| S10 | `80_fig9_markers.py` | label-free check in detail |
| S11 | `57_fig7_robustness.py` | robustness (K, seeds, parameters, leave-one-out) |
| S12 | `83_fig11_guide.py` | scorecard and practical guide in detail |

The November 20, 2026 server export includes [scores](results/), [supplementary tables](results/AdditionalFile1_tables.xlsx), the six [main figures](figures/) and the twelve [supplementary figures](figures/supplementary/). All 18 GraphST runs on the six Wu 2021 sections have an `ok` score record (three seeds per section). The [Wu runner](scripts/70e_fix_wu_py.sh) starts a separate process for each section and method and retries unfinished GraphST seeds on CPU.

Figure 5 and Figure S9 use the 20 breast sections of the external test. Their smoothing-dose panels skip `GSE213688_GSM6592059` and `GSE213688_GSM6592061`, which [Table S2](results/tables/TableS2_sections.csv) excludes because fewer than two classes have 20 or more spots.

## What smoothing corrects and what it costs at a boundary

For 39 labelled sections, we followed each pair of neighbouring spots from no smoothing to one-ring PCA-smooth at neighbour weights of 0.3 and 0.7. We compared the same random seed, counted each undirected pair once and used only pairs with reference labels at both ends. A within-label error splits two neighbours that share a reference label. A boundary error joins neighbours with different reference labels. For each pair type, the error rate after smoothing equals the initial rate minus the fraction corrected plus the fraction newly in error. Each rate uses the full set of eligible pairs of that type.

In the seven sections from one prostatectomy, a weight of 0.3 corrected 6.70% of within-label pairs and introduced errors in 2.24%. At reference boundaries, smoothing corrected 3.30% of pairs and introduced errors in 17.45%. The split rate fell in all seven sections and the boundary merge rate rose in all seven. Each rate averages seeds within a section and sections within a donor. The DLPFC estimates give three donors equal weight. The breast estimates give 20 sections equal weight because the analysis lacks a verified patient key spanning the three breast sources. In the 12 DLPFC sections the same weight raised the donor-averaged ARI from 0.418 to 0.505 while the boundary merge rate rose by 13.81 percentage points, so a better whole-section score can coexist with lost boundaries. The prostate result shows consistency within one specimen; how often it holds across patients needs an independent annotated cohort. [Figure 4d-m](figures/Figure4.png) · [Tables S16 to S18](results/AdditionalFile1_tables.xlsx) · [earlier draft of the passage](results/boundary/Manuscript_insert_Figure_S2.docx) (superseded by the manuscript text and Figure 4)

The [transition counts](results/boundary/paired_edges.csv) retain the original label pair, section, seed and weight, including pairs correct or incorrect at both weights. The [cohort](results/boundary/paired_cohorts.csv), [section](results/boundary/paired_sections.csv) and [prostate label-pair](results/boundary/prostate_label_pairs.csv) tables use rates between 0 and 1. The edge totals in the summaries count each physical pair once per section and comparison. Rates are averaged as described above. The [marginal error tables](results/boundary/) also contain the original 13-method comparison and the dose series.

With a clone of the published result files, the following command rebuilds Tables S16 to S18 and a quick-look plot (`results/boundary/paired_changes.png`); `scripts/m4_fig4.py` draws the published panels. On the original benchmark inputs, `run_all.sh` also reconstructs the spot assignments and checks the paired counts against the baseline and smoothed error totals for every section and seed.

```bash
python scripts/95_paired_summary.py results/boundary
```

## External renal TLS pilot and cohort checks

[External validation](external_validation/) contains spot-level predictions, scores from 27 clustering runs, counts of detected spots in each annotated TLS patch, and a secondary control that shuffles neighbours while preserving each spot's degree. The [maps](external_validation/zurich_tls_maps.png) show seed 0 for all three Zurich kidney tumour sections. The two cohort tables document which HTAN prostate sections carry annotations and which Meylan kidney sections qualify for TLS scoring.

The cohort table script reads the HTAN workbook with pandas and openpyxl. To reproduce these files, put the source data under `external_validation/inputs/` in the locations below.

- Place `KC1.h5ad`, `KC2.h5ad` and `KC3.h5ad` from the archive's `h5ad_preprocessed` directory in `zurich/`. [Zurich Visium data](https://zenodo.org/records/14620362)
- Save the supplementary workbook containing Tables S3 and S12 as `HTAN_supp.xlsx`. [HTAN supplementary tables](https://pmc-oa-opendata.s3.amazonaws.com/PMC13531001.1/cd-26-0012_supplementary_tables_s1-s16_suppst1.xlsx)
- Place the published per-sample `*tissue_positions_list.csv.gz` files and available `*TLS_annotation.csv.gz` files in `GSE175540/`. [Meylan Visium data](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175540)

```bash
python external_validation/cohort_coverage.py
python external_validation/zurich_tls_pilot.py
python external_validation/zurich_tls_pilot.py --shuffle-control
```

The [independent prostate validation plan](external_validation/prostate_validation.md) and [Zurich H&E reader instructions](external_validation/tls_reader_instructions_zh.md) describe follow-up studies.

## Citation

If you use this code, please cite the article (reference to be added on publication) and the original data sets
listed in Table S1.

## License

Code: MIT License (see `LICENSE`). The data belong to their original authors and are subject to their licences.
