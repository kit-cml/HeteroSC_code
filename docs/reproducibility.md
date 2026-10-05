# Default training settings

These are the defaults in the code (`AnnotationConfig`, `DrugResponseConfig`) and the values the tutorials use. They were chosen while developing the method; you retrain from scratch, so your numbers will differ slightly with seeds and hardware. Foundation-model backbones are frozen in every experiment.

## Architecture (all tasks)
Shared dimension $d = 512$. Adapter: `in → max(4d, in/2) → 2d → d` with LayerNorm, GELU, dropout 0.15, Xavier initialisation.
Router: residual MLP on the concatenation (`K·d → 256 → 256 (+residual) → 128 → K`), Xavier gain 0.1 so routing starts near uniform, dropout 0.15.
Every expert embedding is L2-normalised before fusion in the annotation pipeline.

## Cell type annotation
AdamW (weight decay 1e-4), batch 32, 150 epochs maximum, linear warm-up (15 epochs) then cosine decay 5e-4 → 5e-6, gradient clipping 1.0.
Class-weighted cross-entropy (weights ∝ 1/√count, capped at a 5× ratio), label smoothing 0.08, diversity weight λ_div = 0.05.
Mixup (α = 0.4) on a random half of the batches. Early stopping on 0.5·Macro-F1 + 0.5·Accuracy (patience 25); the weights of the three best
checkpoints are averaged before testing. Zero-shot kNN: 5 neighbours, cosine distance, distance-weighted voting.
Predictor head: `512 → 512 → 256 → n_classes` (dropout 0.25).

## Drug response
Batch 128, at most 50 epochs, patience 10, AdamW (lr 1e-4, weight decay 1e-5), 5 warm-up epochs then cosine decay, gradient clipping 1.0, entropy weight λ_H = 0.05 (HeteroSC only).
Early stopping on the validation **per-drug PCC**, where validation = a random 10% of the training drugs. Drug encoders: edge-gated GCN (3 layers, hidden 128, attention pooling),
fingerprint residual MLP (hidden 1024), or their gated ensemble. Prediction head `2d → 256 → 128 → 1`.
Single-expert baselines use a Linear + LayerNorm projection instead of adapters and router.

## Gene perturbation (GEARS)
GEARS hidden size 512, batch 8 (test 32), lr 0.005. Stage 1: GEARS without pretrained gene embeddings (15 epochs). Stage 2: the gene-embedding layer is replaced by
the HeteroSC module and only adapters + router are trained (5 epochs); no additional routing regularisation. Simulation split: seed 3, `train_gene_set_size = 0.75`.
Gene-embedding tables are aligned to the GEARS gene order **by gene identity**; genes missing from a model's vocabulary get N(0,1) vectors.

## Seeds
Annotation split: 42. Drug folds: KFold seed 42. Per-fold training seed: 42 + fold. Perturbation: three model-level seeds with a fixed split.
