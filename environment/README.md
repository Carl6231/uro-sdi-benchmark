# Software environment

Versions used for the article: `software_versions.csv` (Python 3.10, R 4.5.1, CUDA GPU NVIDIA RTX 3080).

## Python

```bash
conda env create -f uro_sdi.yml            # exact environment exported from the analysis machine
# or, from scratch:
conda create -n uro_sdi -c conda-forge python=3.10 scanpy squidpy leidenalg python-igraph scikit-learn \
      pandas matplotlib seaborn statsmodels pip
conda activate uro_sdi
pip install torch==2.7.1 torch_geometric==2.8.1 POT gudhi louvain tqdm h5py scikit-misc
pip install --no-deps SpaGCN==1.2.7 GraphST==1.1.1 SpaceFlow==1.0.4
```

STAGATE (PyG implementation) and SEDR are imported from their source trees, which the scripts expect under
`$URO_SDI_BASE/envs/src/`:

```bash
mkdir -p $URO_SDI_BASE/envs/src && cd $URO_SDI_BASE/envs/src
git clone https://github.com/QIFEIDKN/STAGATE_pyG.git
git clone https://github.com/JinmiaoChenLab/SEDR.git
```

## R

`Rscript install_R_packages.R` installs the R methods into a project library (`$URO_SDI_BASE/envs/Rlib`):
BayesSpace, DR.SC and BANKSY (Bioconductor/CRAN), PRECAST (CRAN), and SPARK and BASS from GitHub.
BASS pins C++11 in `src/Makevars`, which current RcppArmadillo no longer accepts; the script builds it with C++17
(no other change).
