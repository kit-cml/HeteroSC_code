# Method → code

| Method element | Code |
|---|---|
| Expert-specific adapter $f_k$ | `heterosc.modules.AdapterMLP` |
| AdaptiveRouter (concat → residual MLP → softmax) | `heterosc.modules.AdaptiveRouter` |
| Stack + weighted fusion; uniform control ($\alpha=1/K$) | `heterosc.modules.HeteroSCFusion(fusion="adaptive" \| "uniform")` |
| Diversity regulariser $\mathcal L_{div}$ | `heterosc.losses.diversity_loss` (used in `AnnotationTrainer`) |
| Per-input entropy penalty | `heterosc.losses.routing_entropy` (used in `drug_response.train_fold`) |
| Normalised routing entropy | `heterosc.losses.normalized_entropy` |
| Cell annotation, fine-tuned head | `heterosc.annotation` (`HeteroSCClassifier`, `AnnotationTrainer`) |
| Zero-shot kNN | `heterosc.knn.knn_probe` |
| Leave-drug-out folds and guard | `heterosc.drug_response.make_leave_drug_out_folds`, `assert_leave_drug_out` |
| Drug encoders (GCN / fingerprint / ensemble) | `heterosc.drug_response` |
| Gene-level fusion inside GEARS | `heterosc.perturbation.HeteroSCGeneEmbedding`, `attach_heterosc_to_gears` |
| Fusion Ladder (15 combinations) | `heterosc.perturbation.expert_combinations` |
| Frozen embedding extraction | `heterosc.extraction.{geneformer,scgpt,scfoundation,cellfm}` |
| Data download | `heterosc.zenodo`, `heterosc.gdsc` |
