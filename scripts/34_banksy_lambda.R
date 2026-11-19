# 34_banksy_lambda.R
# BANKSY dose-response: benchmark settings (RC normalisation, 2000 HVG, AGF on, k_geom = 18, 20 PCs, mclust G = K),
# only the spatial mixing weight lambda changes; lambda = 0 uses no neighbourhood information at all.
# Output in the benchmark's run format: <runs_dir>/<slide>/banksy-lam<lambda>_k<K>_s<seed>.csv / .json
# Usage: Rscript 34_banksy_lambda.R <slide> <K> <seed> <lam1,lam2,...> <runs_dir>
args <- commandArgs(trailingOnly = TRUE)
slide <- args[1]; K <- as.integer(args[2]); seed <- as.integer(args[3])
lams <- as.numeric(strsplit(args[4], ",")[[1]]); runs_dir <- args[5]
base <- Sys.getenv("URO_SDI_BASE", path.expand("~/uro_sdi_bench"))
.libPaths(c(file.path(base, "envs/Rlib"), .libPaths()))
suppressPackageStartupMessages({library(Matrix); library(jsonlite); library(SummarizedExperiment)
  library(SpatialExperiment); library(Seurat); library(Banksy)})

outd <- file.path(base, runs_dir, slide); dir.create(outd, recursive = TRUE, showWarnings = FALSE)
tag_of <- function(lam) sprintf("banksy-lam%s", format(lam, nsmall = 1))
done <- function(lam) {
  f <- file.path(outd, sprintf("%s_k%d_s%d.json", tag_of(lam), K, seed))
  if (!file.exists(f)) return(FALSE)
  old <- tryCatch(fromJSON(f), error = function(e) NULL)
  !is.null(old) && identical(old$status, "ok")
}
lams <- lams[!vapply(lams, done, logical(1))]
if (length(lams) == 0) { cat("skip (done):", slide, "\n"); quit(save = "no") }

ind <- file.path(base, "data/r_input", slide)
cnt <- as(readMM(file.path(ind, "counts.mtx")), "CsparseMatrix")
genes <- readLines(file.path(ind, "genes.txt")); bc <- readLines(file.path(ind, "barcodes.txt"))
rownames(cnt) <- make.unique(genes); colnames(cnt) <- bc
co <- read.csv(file.path(ind, "coords.csv"), row.names = 1, check.names = FALSE)
stopifnot(identical(rownames(co), bc))

set.seed(seed); t0 <- Sys.time()
seu <- CreateSeuratObject(counts = cnt)
seu <- NormalizeData(seu, normalization.method = "RC", scale.factor = median(colSums(cnt)), verbose = FALSE)
seu <- FindVariableFeatures(seu, nfeatures = 2000, verbose = FALSE)
nc <- GetAssayData(seu, layer = "data")[VariableFeatures(seu), ]
se0 <- SpatialExperiment(assays = list(normcounts = nc), spatialCoords = as.matrix(co[, c("x", "y")]))
se0 <- computeBanksy(se0, assay_name = "normcounts", compute_agf = TRUE, k_geom = 18, verbose = FALSE)
t_pre <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

for (lam in lams) {
  tag <- tag_of(lam); stem <- file.path(outd, sprintf("%s_k%d_s%d", tag, K, seed))
  t1 <- Sys.time(); status <- "ok"; err <- ""; pred <- NULL
  tryCatch({
    se <- runBanksyPCA(se0, use_agf = TRUE, lambda = lam, npcs = 20, seed = seed)
    se <- clusterBanksy(se, use_agf = TRUE, lambda = lam, algo = "mclust", mclust.G = K, seed = seed)
    pred <- as.integer(colData(se)[[clusterNames(se)[1]]])
    stopifnot(length(pred) == length(bc), !any(is.na(pred)))
  }, error = function(e) { status <<- "fail"; err <<- conditionMessage(e) })
  sec <- t_pre + as.numeric(difftime(Sys.time(), t1, units = "secs"))
  rec <- list(slide = slide, method = tag, k = K, seed = seed, status = status, sec = round(sec, 1),
              n_spots = length(bc), gpu_peak_mb = 0, error = err, lambda = lam)
  if (status == "ok") {
    write.csv(data.frame(spot = bc, cluster = pred), paste0(stem, ".csv"), row.names = FALSE, quote = FALSE)
    rec$n_found <- length(unique(pred))
  }
  writeLines(toJSON(rec, auto_unbox = TRUE), paste0(stem, ".json"))
  cat(sprintf("%s %s k=%d %s %.1fs %s\n", substr(slide, nchar(slide) - 21, nchar(slide)), tag, K, status, sec, substr(err, 1, 200)))
}
