# Independent prostate validation

This fixed comparison is for a new annotated prostate cohort. The Erickson patient and the breast sections already used for development or evaluation are excluded from this test.

## Cohort and inputs

The first candidate is ProstOmics, with 32 Visium tissue samples from eight prostate cancer patients. Data access is controlled by NTNU. The requested inputs are filtered counts, tissue coordinates, a de-identified patient-to-section key and the original pathologist's spot labels. A patient represented by several samples counts once. [Dataset and access policy](https://ega-archive.org/datasets/EGAD50000000603)

Before scoring, match section and barcode identifiers and map the source's pathology dictionary to malignant epithelium, benign glandular epithelium and stroma. Preserve the source grades separately. Ambiguous or mixed labels enter the primary comparison only if the original dictionary unambiguously assigns one of these three compartments. Record any unmapped category. Other annotated tissue can be reported separately. Unassigned spots are excluded from scoring and retained as input to clustering.

## Comparison fixed before receiving test labels

Fit each section using expression and spatial coordinates. Use the repository's common preprocessing and a fixed K of 3 for the three prespecified compartments. The primary comparison is one-ring PCA-smooth with neighbour weight 0.3 versus weight 0 on the same PCs. Use the same PCA, mixture settings and seeds 0, 1 and 2 for both weights. Run GraphST with the same K and seeds as a secondary comparator. The previously examined strong weight of 0.7 is a sensitivity analysis. Hold these choices fixed across patients. Pathology labels enter scoring after predictions are frozen.

For sections with at least two scored reference compartments, compute ARI for each seed on the same scored spots. Average seeds within section and sections within patient. Report the paired difference for every patient, the equal-patient mean, the number with positive and negative differences, and how much the mean changes when each patient is left out. Single-compartment sections can contribute to within-compartment edge estimates where available. Report distinct denominators for each endpoint.

Secondary endpoints are the fraction of within-compartment neighbour pairs split and the fraction of cross-compartment pairs merged. Build an undirected first-ring spatial graph before excluding unscored pairs. Report both numerators and denominators, including unavailable endpoints when one pair type is absent. A method or seed missing after execution remains missing. Compare methods within their shared scored sections and report coverage. Clinical outcomes are outside this validation.

The open Apostolov Visium deposit includes count matrices and H&E. Its authors' GitHub records a DVC pointer for a reviewed spot annotation directory. The tissue images, counts, patient mapping and actual pathology labels must be matchable before including an independent section. Separate the in-house specimens from re-used published sections. [Visium deposit](https://zenodo.org/records/17411292) · [Authors' analysis](https://github.com/swarbricklab/pca_visium_analysis)
