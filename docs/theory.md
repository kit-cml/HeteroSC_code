# Theory in five minutes

## Why combine foundation models?
Single-cell foundation models (ScFMs) differ in architecture, pre-training objective, input encoding, gene vocabulary and training data.
Benchmarks show that no single model wins everywhere, so their representations are not interchangeable and may be **complementary**.
HeteroSC asks whether they can be combined *adaptively* instead of picking one or averaging them.

## The model
Let $e_i^k \in \mathbb{R}^{d_k}$ be the frozen embedding that expert $k \in \{1..K\}$ produces for input $i$
(a cell, a gene or a cell line, depending on the task). The experts have different sizes $d_k$ and live in different spaces.

1. **Expert-specific adapters** map every expert into a common space $\mathbb{R}^d$:
   $h_i^k = f_k(e_i^k)$ (three linear layers with LayerNorm, GELU and dropout).
2. **AdaptiveRouter** looks at all adapted representations and decides how much to trust each expert for this input:
   $c_i = [h_i^1;\dots;h_i^K]$, $\; z_i = \mathrm{MLP}(c_i) \in \mathbb{R}^K$, $\; \alpha_{ik} = \dfrac{\exp z_{ik}}{\sum_j \exp z_{ij}}$.
3. **Weighted fusion**: $h_i = \sum_{k} \alpha_{ik}\, h_i^k$, which feeds a small task head.

Only adapters, router and head are trained. The foundation-model backbones are never fine-tuned, and adding a new expert only needs a new adapter.

The **uniform-fusion control** replaces the router by $\alpha_{ik} = 1/K$ and keeps everything else identical. Comparing it with HeteroSC isolates
the contribution of *routing* from the contribution of simply *having several representations*.

Relation to Mixture of Experts: this is a *dense, soft* mixture over pre-trained experts, with no sparsity or top-$k$ selection and no joint training of the experts.

## What is the routing unit?
| Task | Router decides per | Objective |
|---|---|---|
| Cell type annotation | cell | class-weighted cross-entropy (label smoothing) + diversity term |
| Gene perturbation (GEARS) | gene | the GEARS objective only |
| Unseen-drug response | cell line | MSE on AUC + per-input entropy penalty |

## Routing regularisers
* **Diversity** (annotation): with $\bar\alpha_k$ the batch-mean routing weight,
  $\mathcal{L}_{div} = \sum_k \bar\alpha_k \log \bar\alpha_k$ (negative entropy of the *batch-mean* distribution). Minimising it discourages the router
  from collapsing onto one expert across a batch, while leaving per-cell routing free.
* **Entropy penalty** (drug response): $\mathcal{L} = \mathcal{L}_{MSE} + \lambda_H \tfrac{1}{N}\sum_i H(\alpha_i)$ with $H(\alpha_i) = -\sum_k \alpha_{ik}\log\alpha_{ik}$.
  It acts on each input and encourages selective expert allocation, the opposite pressure of the diversity term (it acts per input, not on the batch mean).

## Routing diagnostics
* **Normalised entropy** of the mean routing distribution, $H_{norm} = -\sum_k \bar\alpha_k\ln\bar\alpha_k / \ln K \in [0,1]$: 1 means even usage, 0 means one expert dominates.
* **Mean routing weight per expert** per dataset or group.

Routing weights are *predictive contributions under the training objective*. A regulariser changes them, and an expert with a low weight is not
thereby a "worse" model. They are not evidence of biological specialisation.

## Validation protocols
* Annotation: stratified train/validation/test split shared by all experts; fine-tuned head and a zero-shot kNN probe (5 neighbours, cosine, distance-weighted).
* Drug response: **leave-drug-out** cross-validation. Every drug is in exactly one fold, so test drugs are never seen in training.
* Perturbation: the GEARS *simulation* split, which holds out genes.
