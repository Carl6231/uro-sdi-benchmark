#!/usr/bin/env bash
# run_all.sh - the complete analysis in the order it was run. Every step writes to $URO_SDI_BASE (data/, results/, logs/).
# The steps are long (the whole benchmark took several days on one workstation with one GPU); they can be run one at a
# time, and the method runs (steps 3-5) can be split over several processes with the --slides/--seeds options.
set -euo pipefail
export URO_SDI_BASE=${URO_SDI_BASE:-$HOME/uro_sdi_bench}
REPO=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$URO_SDI_BASE"/{logs,meta,data,results}
ln -sfn "$REPO/scripts" "$URO_SDI_BASE/scripts"
cd "$URO_SDI_BASE"
PY=${PY:-python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

# ---- 1. data -------------------------------------------------------------------------------------------------------
$PY scripts/01_inventory_stimage.py                  # file list of STimage-1K4M
$PY scripts/02_inspect_annotations.py                # which sections carry pathologist annotation
$PY scripts/03_download_stimage.py                   # prostate, kidney, ccRCC and 10x sections
$PY scripts/03b_download_dlpfc.py                    # 12 DLPFC sections
$PY scripts/06_build_h5ad.py ALL 8                   # one .h5ad per section: counts, spot positions, labels
$PY scripts/62_make_sim.py                           # 36 simulated sections (Gaussian random-field layouts)
$PY scripts/67_get_breast.py                         # breast-cancer sections (GSE213688, 10x)
$PY scripts/06_build_h5ad.py GSE213688 4
$PY scripts/06_build_h5ad.py Human_Breast_10X_06 4
$PY scripts/69_fix_breast_labels.py                  # harmonise the breast labels
# Wu et al. 2021 (Zenodo 4739739): put filtered_count_matrices.tar.gz, spatial.tar.gz and metadata.tar.gz in
# data/raw_ext/wu2021/, then:
$PY scripts/68_build_wu2021.py

# ---- 2. inputs for the R methods ------------------------------------------------------------------------------------
$PY scripts/14_export_for_r.py "" 4                  # all sections
$PY scripts/16_make_r_jobs.py                        # meta/r_jobs.txt: section, method, K, seed

# ---- 3. Python methods (GraphST, STAGATE, SEDR, SpaGCN, SpaceFlow, PCA-smooth, GMM, Leiden) --------------------------
$PY scripts/08_run_methods.py --slides Erickson,Maynard --seeds 0,1,2,3,4            # labelled, K = annotated classes
$PY scripts/08_run_methods.py --slides GSE175540 --k 6 --seeds 0                     # ccRCC (TLS task)
$PY scripts/08_run_methods.py --slides GSE175540 --k 10 --seeds 0
$PY scripts/08_run_methods.py --slides GSE183456,GSE206306,GSE226533,Human_Kidney_10X,Human_Prostate_10X --k 6 --seeds 0
$PY scripts/08_run_methods.py --slides SIM_ --seeds 0,1,2 --runs results/runs_sim
$PY scripts/08_run_methods.py --slides GSE213688,Human_Breast_10X_06092021_Visium,Human_Breast_10X_06232020_Visium_Block_A_Section_1 \
    --seeds 0,1,2 --runs results/runs_ext
PY="$PY" bash scripts/70e_fix_wu_py.sh

# ---- 4. R methods (BayesSpace, BANKSY, DR-SC, PRECAST, BASS): one job per line "section method K seed [runs_dir]" -------
xargs -P 4 -L 1 Rscript scripts/15_run_r_methods.R < meta/r_jobs.txt
for queue in "$REPO"/jobs/r_jobs_*.txt; do
  xargs -P 4 -L 1 Rscript scripts/15_run_r_methods.R < "$queue"
done

# ---- 5. sensitivity analyses -----------------------------------------------------------------------------------------
$PY scripts/33_dose_smooth.py 4                       # PCA-smooth: neighbour weight alpha x rings
$PY scripts/63_dose_generic.py Erickson,Maynard,GSE175540,GSE183456,GSE206306,GSE226533,Human_Kidney_10X,Human_Prostate_10X real 4 1 0,1,2
$PY scripts/63_dose_generic.py SIM_ sim 4 1,2,3 0,1,2
$PY scripts/63_dose_generic.py GSE213688,Human_Breast_10X_06092021_Visium,Human_Breast_10X_06232020_Visium_Block_A_Section_1,WU2021_ ext 3 1 0,1,2
$PY scripts/35_stagate_radius.py --slides Erickson,Maynard --factors 2.1,3.1 --seeds 0,1,2
xargs -P 4 -L 1 Rscript scripts/34_banksy_lambda.R < "$REPO/jobs/banksy_lambda_jobs.txt"
$PY scripts/08_run_methods.py --runs results/runs_ksens --slides Erickson --seeds 0 --kdelta=-2,-1,1,2   # K sweep

# ---- 6. scoring, statistics and tables -------------------------------------------------------------------------------
$PY scripts/09_score.py
$PY scripts/09_score.py results/runs_sim results/scores_sim.csv
$PY scripts/09_score.py results/runs_ext results/scores_ext.csv
$PY scripts/09_score.py results/runs_ksens results/scores_ksens.csv
$PY scripts/09_score.py results/runs_dose results/scores_dose.csv
$PY scripts/20_stats.py
$PY scripts/36_robustness.py
$PY scripts/43_label_free_diagnostic.py
$PY scripts/45_dose_ksens_stats.py
$PY scripts/47_fixedk_pas.py 3
$PY scripts/65_markers_main.py results/runs marker_main 4
$PY scripts/65_markers_main.py results/runs_ext marker_ext 4
$PY scripts/74_marker_grain.py 4
$PY scripts/77_expression_coherence.py 4
$PY scripts/72_followup_tables.py
$PY scripts/78_table_datasets.py

# ---- 7. figures ------------------------------------------------------------------------------------------------------
for s in 51_fig1_overview 52_fig2_accuracy 53_fig3_transfer_maps 54_fig4_tls 55_fig5_mechanism 56_fig6_dose \
         81_fig7_simulation 82_fig8_external 80_fig9_markers 57_fig7_robustness 83_fig11_guide; do
  $PY scripts/$s.py
done
$PY scripts/91_additional_file1.py                   # Additional file 1: all supplementary tables in one Excel workbook (needs openpyxl)
echo "done: figures in $URO_SDI_BASE/results/figures_full, tables in $URO_SDI_BASE/results"
