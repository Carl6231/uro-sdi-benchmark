# install_R_packages.R - R methods used in the benchmark, installed into a project library.
# Usage: Rscript install_R_packages.R      (needs R >= 4.5, a C++17 compiler, git and an internet connection)
base <- Sys.getenv("URO_SDI_BASE", path.expand("~/uro_sdi_bench"))
lib <- file.path(base, "envs", "Rlib"); dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4, timeout = 900)
have <- function(p) p %in% rownames(installed.packages())
if (!have("BiocManager")) install.packages("BiocManager", lib = lib)
bioc <- c("BayesSpace", "DR.SC", "Banksy", "SingleCellExperiment", "SpatialExperiment", "scater", "scran")
cran <- c("Seurat", "mclust", "Matrix", "jsonlite", "PRECAST", "GIGrvg", "label.switching", "CompQuadForm", "matlab",
          "pracma", "doParallel", "foreach", "harmony", "RcppDist", "RcppArmadillo", "remotes")
for (p in cran) if (!have(p)) install.packages(p, lib = lib)
for (p in bioc) if (!have(p)) BiocManager::install(p, lib = lib, update = FALSE, ask = FALSE)

src <- file.path(base, "envs", "src"); dir.create(src, showWarnings = FALSE)
gh <- function(repo, dir) {
  if (!dir.exists(file.path(src, dir))) system2("git", c("clone", "--depth", "1", paste0("https://github.com/", repo, ".git"), file.path(src, dir)))
}
if (!have("SPARK")) { gh("xzhoulab/SPARK", "SPARK"); install.packages(file.path(src, "SPARK"), repos = NULL, type = "source", lib = lib) }
if (!have("BASS")) {
  gh("zhengli09/BASS", "BASS")
  for (f in list.files(file.path(src, "BASS", "src"), pattern = "^Makevars", full.names = TRUE))
    writeLines(sub("CXX_STD *= *CXX11", "CXX_STD = CXX17", readLines(f)), f)     # C++11 -> C++17, nothing else changed
  install.packages(file.path(src, "BASS"), repos = NULL, type = "source", lib = lib)
}
for (p in c("BayesSpace", "Banksy", "DR.SC", "PRECAST", "SPARK", "BASS"))
  cat(sprintf("%-11s %s\n", p, if (have(p)) as.character(packageVersion(p, lib.loc = lib)) else "MISSING"))
