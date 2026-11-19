"""marker_programs.py - curated marker-gene programmes used for the label-free biological check.

A programme score is the mean of per-gene z-scores of log(CP10K + 1) expression. For each programme we also describe
the spatial grain of its high-expression region: the boundary share of the top-decile spots (share of top-decile
spots that touch a spot outside the top decile on the first lattice ring) and Moran's I of the score.
The domain-level agreement of a clustering with a programme is eta^2 = between-domain variance / total variance.
"""
import numpy as np, scipy.sparse as sp

PROSTATE = {
    "Prostate tumour": ["AMACR", "HPN", "GOLM1", "OR51E2", "SPON2", "PCA3", "TDRD1", "CRISP3", "DLX1", "FOLH1"],
    "Benign glands": ["MSMB", "KRT5", "KRT15", "TP63"],
    "Smooth-muscle stroma": ["ACTA2", "MYH11", "DES", "CNN1", "TAGLN", "ACTG2"],
    "Lymphocytes": ["PTPRC", "CD3E", "CD2", "MS4A1", "CD79A"],
}
KIDNEY = {
    "Glomeruli (podocytes)": ["NPHS1", "NPHS2", "PODXL", "PTPRO", "CLIC5", "WT1", "PLA2R1"],
    "Proximal tubule": ["LRP2", "CUBN", "SLC34A1", "SLC5A12", "SLC22A6", "SLC13A3"],
    "Thick ascending limb": ["UMOD", "SLC12A1"],
    "Collecting duct": ["AQP2", "AQP3", "FXYD4", "SCNN1G"],
    "Endothelium": ["PECAM1", "VWF", "EMCN", "PLVAP"],
    "Lymphocytes": ["PTPRC", "CD3E", "CD2", "MS4A1", "CD79A"],
}
CCRCC = dict(KIDNEY, **{
    "TLS (B cells, FDC)": ["MS4A1", "CD79A", "CD19", "CR2", "FCER2", "CXCL13", "LTB", "CCL19", "CCL21"],
    "ccRCC tumour (hypoxia)": ["CA9", "NDUFA4L2", "ANGPTL4", "EGLN3", "VEGFA"],
})
DLPFC = {
    "White matter": ["MBP", "PLP1", "MOBP", "MOG", "CNP"],
    "Neurons": ["SNAP25", "NEFL", "SYT1", "STMN2"],
    "Layer 1 (astrocytes)": ["AQP4", "GFAP", "RELN"],
}
BREAST = {
    "Epithelium": ["KRT8", "KRT18", "KRT19", "EPCAM", "CDH1"],
    "Myoepithelium": ["KRT5", "KRT14", "TP63", "KRT17"],
    "Fibroblast stroma": ["COL1A1", "COL1A2", "DCN", "LUM", "FBLN1"],
    "Lymphocytes": ["PTPRC", "CD3E", "CD2", "MS4A1", "CD79A"],
    "Adipocytes": ["ADIPOQ", "PLIN1", "LEP", "FABP4"],
}


def programs_for(sid):
    if "Erickson" in sid or "Prostate" in sid:
        return "prostate", PROSTATE
    if "GSE175540" in sid:
        return "ccRCC", CCRCC
    if any(p in sid for p in ("GSE183456", "GSE206306", "GSE226533", "Kidney")):
        return "kidney", KIDNEY
    if "Maynard" in sid:
        return "DLPFC", DLPFC
    if sid.startswith("WU2021") or sid.startswith("HER2ST") or "GSE213688" in sid or "Breast" in sid:
        return "breast", BREAST
    return None, {}


def lognorm(a):
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    tot = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(tot, 1)) @ X
    X.data = np.log1p(X.data)
    return X.tocsc()


def program_scores(a, progs, min_frac=0.01):
    """Dict programme -> (score vector, genes used). Genes must be detected in >= min_frac of spots; >= 2 genes needed."""
    X = lognorm(a)
    name_to_j = {g: j for j, g in enumerate(a.var_names)}
    out = {}
    for p, genes in progs.items():
        cols = []
        used = []
        for g in genes:
            j = name_to_j.get(g)
            if j is None:
                continue
            v = X[:, j].toarray().ravel()
            if (v > 0).mean() < min_frac or v.std() == 0:
                continue
            cols.append((v - v.mean()) / v.std()); used.append(g)
        if len(cols) >= 2:
            out[p] = (np.mean(cols, axis=0), used)
    return out


def eta2(score, labels):
    s = np.asarray(score, float); lab = np.asarray(labels)
    mu = s.mean(); tot = ((s - mu) ** 2).sum()
    if tot == 0:
        return np.nan
    btw = sum(((lab == c).sum()) * (s[lab == c].mean() - mu) ** 2 for c in np.unique(lab))
    return float(btw / tot)


def grain(score, nb, q=0.9):
    """(boundary share of the top-decile spots, Moran's I with first-ring weights)."""
    s = np.asarray(score, float)
    top = s >= np.quantile(s, q)
    bd = np.array([top[i] and (len(n) > 0) and (~top[n]).any() for i, n in enumerate(nb)])
    boundary = float(bd[top].mean()) if top.any() else np.nan
    z = s - s.mean()
    num = sum(z[i] * z[n].sum() for i, n in enumerate(nb))
    W = sum(len(n) for n in nb)
    moran = float(len(s) / W * num / (z ** 2).sum()) if W and (z ** 2).sum() > 0 else np.nan
    return boundary, moran


def top_patch(score, nb, q=0.9):
    """Spot-weighted mean size of the connected patches (first lattice ring) formed by the top-decile spots:
    a few spots for scattered small structures, hundreds for large compartments."""
    from scipy.sparse.csgraph import connected_components
    s = np.asarray(score, float)
    top = np.where(s >= np.quantile(s, q))[0]
    pos = -np.ones(len(s), dtype=int); pos[top] = np.arange(len(top))
    r, c = [], []
    for i in top:
        for j in nb[i]:
            if pos[j] >= 0:
                r.append(pos[i]); c.append(pos[j])
    G = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(top), len(top)))
    _, comp = connected_components(G, directed=False)
    size = np.bincount(comp)[comp]
    return float(size.mean())
