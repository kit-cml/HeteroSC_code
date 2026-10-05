# Data

Everything needed to run the tutorials except the foundation-model checkpoints is in one Zenodo record:
**https://doi.org/10.5281/zenodo.23153537** (three `.rar` archives, about 650 MB). `heterosc.zenodo.download_data()` downloads, verifies the MD5 and extracts them.

| Archive | Contents |
|---|---|
| `annotation` | `raw/` original `.h5ad` (Pancreas, Skin, Liver, Lung, Myeloid) · `processed/` `.h5ad` with `cell_type`, `var['ensembl_id']`, `n_counts`, `obs['master_split']` · `splits/` `cell_id → split` tables · `annotation_split_summary.csv` |
| `drug_response` | `response_table_merged_leave_drug_out.csv` (`cell_line_id, drug_id, canonical_smiles, auc, fold`) · `drug_fold_assignment.csv` · `drug_graph_features_merged.pkl` · `drug_fingerprint_merged.csv` (MACCS 167 + ECFP4 2048) · `cellline_expression.h5ad` · `gdsc_drugname_to_smiles_cache.json` · `raw/DepMap_Model.csv` |
| `gene_perturbation` | `norman.zip`, `adamson.zip` (GEARS PertData) · source `.h5ad` · gene-symbol→Ensembl tables · GEARS simulation split (`*_simulation_3_0.75*.pkl`) · GEARS GO resources |

Archive MD5: `annotation.rar` 0cb08bcad13e54b0b42492b929bbd097 · `drug_response.rar` 43fec0e20632007264c238c7a04029f5 · `gene_perturbation.rar` 7b50c606d74eb7dddc499a882eb5ee74.

Opening `.rar` needs 7-Zip (`7z`), `unrar`, or `pip install rarfile` plus unrar. `.pkl` files should only be loaded from trusted sources.

## Not in the record
* **Foundation-model checkpoints and embeddings.** Download the checkpoints from the official releases (links in the README) and extract embeddings with notebooks `02a–d`.
* **Raw GDSC dose-response files.** Download from the GDSC website under its terms (`scripts/download_data.py --gdsc data/gdsc`). The processed response table in the record is enough for the tutorials.

## Protocols encoded in the data
* **Annotation:** cell types with fewer than 10 cells removed; stratified 70/15/15 train/validation/test, random seed 42, shared by all four models. Only cells with a valid embedding from all four models are used.
* **Drug response:** 109,152 cell line–drug pairs, 223 drugs, 551 cell lines (the expression file holds 561 profiles; 551 have drug-response data), target = AUC from GDSC1 + GDSC2 (GDSC2 takes precedence for overlapping pairs). `fold` = 10-fold **leave-drug-out**: KFold over the sorted drug list, seed 42, 22–23 drugs per fold.
* **Perturbation:** Norman and Adamson processed by GEARS; simulation split with seed 3 and `train_gene_set_size = 0.75`.

## Embedding sizes
| Expert | Cell embedding | Source |
|---|---|---|
| Geneformer V2-104M | 768 | CLS token, last hidden layer |
| scGPT whole-human | 512 | CLS embedding (L2-normalised) |
| CellFM (80M) | 1536 | CLS token |
| scFoundation | 768 | max-pool over encoder tokens |

## Sources and attribution
Please cite the original data providers: Cheng et al. 2021 (Myeloid), MacParland et al. 2018 (Liver), the CellFM benchmark data (Zeng et al. 2024),
Norman et al. 2019, Adamson et al. 2016, Roohani et al. 2024 (GEARS), Yang et al. 2013 and Iorio et al. 2016 (GDSC), Barretina et al. 2012 and Ghandi et al. 2019 (CCLE).
GDSC-derived values follow the [GDSC data usage policy](https://depmap.sanger.ac.uk/documentation/data-usage-policy/); DepMap data are generally CC BY 4.0.
