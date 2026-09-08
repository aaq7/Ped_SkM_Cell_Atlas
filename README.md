# Pediatric MuSCs

Analysis of pediatric skeletal muscle satellite cells (MuSCs) from the BCH cohort.

## Pipeline

- `seurat_export.Rmd` — exports the Seurat object for AnnData object creation
- `mrvi_subcluster.py` — subclusters satellite cells with MrVI
- `mrvi_plots.py` — plots the MrVI subclustering
- `age_characterization.py` — age-related abundance/expression analysis
- `cornell_comparison.py` — compares subclusters to the Cornell atlas' MuSC/progenitor states
- `pseudotimekernel_trajectory.py` — CellRank trajectory using DPT rooted at a quiescent PAX7+ cell
- `cytotracekernel_trajectory.py` — CellRank trajectory using CytoTRACE
- `utils.py` — shared paths, params, palettes, helpers

Figures land in `figures/` and `results_age/`, tables in `tables/`.

## Notes

- `leiden_musc` at resolution 0.3 is the subtype clustering used everywhere (`SUBTYPE_ORDER`, `cluster_label`)
- age groups are binned per pediatric convention: Infant (0-1), Early Childhood (1-5), Late Childhood (5-11), Adolescence (11+)
