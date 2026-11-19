# 15_run_r_methods.R
# Run one R spatial-domain method on one exported slide; output format matches 08_run_methods.py.
# Usage: Rscript 15_run_r_methods.R <slide> <bayesspace|drsc|banksy> <K> <seed>
args <- commandArgs(trailingOnly = TRUE)
slide <- args[1]; method <- args[2]; K <- as.integer(args[3]); seed <- as.integer(args[4])
base <- Sys.getenv("URO_SDI_BASE", path.expand("~/uro_sdi_bench"))
.libPaths(c(file.path(base, "envs/Rlib"), .libPaths()))
suppressPackageStartupMessages({library(Matrix); library(jsonlite)})

runs_dir <- if (length(args) >= 5) args[5] else "results/runs"     # optional 5th argument: output folder (sensitivity runs)
outd <- file.path(base, runs_dir, slide); dir.create(outd, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(outd, sprintf("%s_k%d_s%d", method, K, seed))
if (file.exists(paste0(stem, ".json"))) {
  old <- tryCatch(fromJSON(paste0(stem, ".json")), error = function(e) NULL)
  if (!is.null(old) && identical(old$status, "ok")) { cat("skip (done):", slide, method, "\n"); quit(save = "no") }
}

ind <- file.path(base, "data/r_input", slide)
cnt <- as(readMM(file.path(ind, "counts.mtx")), "CsparseMatrix")          # genes x spots, raw counts
genes <- readLines(file.path(ind, "genes.txt")); bc <- readLines(file.path(ind, "barcodes.txt"))
rownames(cnt) <- make.unique(genes); colnames(cnt) <- bc
co <- read.csv(file.path(ind, "coords.csv"), row.names = 1, check.names = FALSE)
stopifnot(identical(rownames(co), bc))

# Spots left with zero counts after gene filtering break size-factor normalisation (BayesSpace, DR-SC).
# Those methods are run without such spots; each dropped spot then gets the label of its nearest retained spot
# (same rule as for SpaGCN in the Python runner).
keep <- Matrix::colSums(cnt) > 0
fill_nearest <- function(pred_kept) {
  full <- integer(length(bc)); full[keep] <- pred_kept
  if (any(!keep)) {
    xk <- co$x[keep]; yk <- co$y[keep]
    for (i in which(!keep)) full[i] <- pred_kept[which.min((xk - co$x[i])^2 + (yk - co$y[i])^2)]
  }
  full
}

set.seed(seed); t0 <- Sys.time(); status <- "ok"; err <- ""; pred <- NULL
tryCatch({
  if (method == "bayesspace") {
    # BayesSpace vignette settings for Visium (t-distributed errors, gamma = 3); 10,000 MCMC iterations
    suppressPackageStartupMessages({library(SingleCellExperiment); library(BayesSpace)})
    ck <- co[keep, ]
    cd <- DataFrame(row = ck$array_row, col = ck$array_col, array_row = ck$array_row, array_col = ck$array_col,
                    imagerow = ck$y, imagecol = ck$x, pxl_row_in_fullres = ck$y, pxl_col_in_fullres = ck$x, row.names = bc[keep])
    sce <- SingleCellExperiment(assays = list(counts = cnt[, keep]), colData = cd)
    sce <- spatialPreprocess(sce, platform = "Visium", n.PCs = 15, n.HVGs = 2000, log.normalize = TRUE)
    sce <- spatialCluster(sce, q = K, platform = "Visium", d = 15, init.method = "mclust", model = "t",
                          gamma = 3, nrep = 10000, burn.in = 1000, save.chain = FALSE)
    pred <- fill_nearest(as.integer(sce$spatial.cluster))
  } else if (method == "drsc") {
    # DR-SC tutorial: log-normalise, 2000 variable genes, Visium neighbourhood from array row / col
    suppressPackageStartupMessages({library(Seurat); library(DR.SC)})
    seu <- CreateSeuratObject(counts = cnt[, keep], meta.data = data.frame(row = co$array_row[keep], col = co$array_col[keep], row.names = bc[keep]))
    seu <- NormalizeData(seu, verbose = FALSE)
    seu <- FindVariableFeatures(seu, nfeatures = 2000, verbose = FALSE)
    seu <- DR.SC(seu, K = K, platform = "Visium", verbose = FALSE)
    pred <- fill_nearest(as.integer(seu$spatial.drsc.cluster))
  } else if (method == "banksy") {
    # BANKSY domain-segmentation settings: lambda = 0.2, AGF on, k_geom = 18, 20 PCs; mclust with G = K
    suppressPackageStartupMessages({library(SummarizedExperiment); library(SpatialExperiment); library(Seurat); library(Banksy)})
    seu <- CreateSeuratObject(counts = cnt)
    seu <- NormalizeData(seu, normalization.method = "RC", scale.factor = median(colSums(cnt)), verbose = FALSE)
    seu <- FindVariableFeatures(seu, nfeatures = 2000, verbose = FALSE)
    hv <- VariableFeatures(seu)
    nc <- GetAssayData(seu, layer = "data")[hv, ]
    se <- SpatialExperiment(assays = list(normcounts = nc), spatialCoords = as.matrix(co[, c("x", "y")]))
    se <- computeBanksy(se, assay_name = "normcounts", compute_agf = TRUE, k_geom = 18, verbose = FALSE)
    se <- runBanksyPCA(se, use_agf = TRUE, lambda = 0.2, npcs = 20, seed = seed)
    se <- clusterBanksy(se, use_agf = TRUE, lambda = 0.2, algo = "mclust", mclust.G = K, seed = seed)
    pred <- as.integer(colData(se)[[clusterNames(se)[1]]])
  } else if (method == "bass") {
    # BASS tutorial settings for one section: C = 20 cell types, R = K domains, 3,000 SPARK-X genes, 20 PCs (unscaled),
    # mclust initialisation, Swendsen-Wang update of the interaction parameter, 10,000 posterior samples (burn-in 2,000)
    suppressPackageStartupMessages({library(BASS)})
    xyk <- as.matrix(co[keep, c("x", "y")]); rownames(xyk) <- bc[keep]
    bo <- createBASSObject(list(cnt[, keep]), list(xyk), C = 20, R = K, beta_method = "SW", init_method = "mclust", nsample = 10000)
    bo <- BASS.preprocess(bo, doLogNormalize = TRUE, geneSelect = "sparkx", nSE = 3000, doPCA = TRUE, scaleFeature = FALSE, nPC = 20)
    bo <- BASS.run(bo)
    bo <- BASS.postprocess(bo)
    pred <- fill_nearest(as.integer(bo@results$z[[1]]))
  } else if (method == "precast") {
    # PRECAST vignette settings for one section: 2,000 SPARK-X genes, Visium neighbourhood, K fixed, maxIter = 30
    suppressPackageStartupMessages({library(Seurat); library(PRECAST)})
    seu <- CreateSeuratObject(counts = cnt[, keep], meta.data = data.frame(row = co$array_row[keep], col = co$array_col[keep], row.names = bc[keep]))
    po <- CreatePRECASTObject(list(seu), gene.number = 2000, selectGenesMethod = "SPARK-X", verbose = FALSE)
    po <- AddAdjList(po, platform = "Visium")
    po <- AddParSetting(po, Sigma_equal = FALSE, maxIter = 30, verbose = FALSE, coreNum = 1, seed = seed)
    po <- PRECAST(po, K = K)
    po <- SelectModel(po)
    kept <- colnames(po@seulist[[1]])
    cl <- as.integer(po@resList$cluster[[1]])
    # spots removed by PRECAST's quality filter get the label of their nearest retained spot
    ik <- match(kept, bc); full <- integer(length(bc)); full[ik] <- cl
    for (i in setdiff(seq_along(bc), ik)) full[i] <- cl[which.min((co$x[ik] - co$x[i])^2 + (co$y[ik] - co$y[i])^2)]
    pred <- full
  } else stop("unknown method: ", method)
  stopifnot(length(pred) == length(bc), !any(is.na(pred)))
}, error = function(e) { status <<- "fail"; err <<- conditionMessage(e) })

sec <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
rec <- list(slide = slide, method = method, k = K, seed = seed, status = status, sec = round(sec, 1),
            n_spots = length(bc), gpu_peak_mb = 0, error = err)
if (status == "ok") {
  write.csv(data.frame(spot = bc, cluster = pred), paste0(stem, ".csv"), row.names = FALSE, quote = FALSE)
  rec$n_found <- length(unique(pred))
}
writeLines(toJSON(rec, auto_unbox = TRUE), paste0(stem, ".json"))
cat(sprintf("%s %s k=%d s=%d %s %.1fs %s\n", substr(slide, nchar(slide) - 21, nchar(slide)), method, K, seed, status, sec, substr(err, 1, 200)))
