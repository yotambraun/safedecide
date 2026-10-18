# Method

## 1. Setting

A decision is a state *x*, a question, and a closed set of options
{1..K}. The model returns a distribution *p(·|x)*. We act on the top option
*ŷ = argmax p* with confidence *c(x) = max p*.

The deployment question is binary per decision: **automate** (act on ŷ) or
**escalate** (send to a human or a bigger model). A threshold policy automates
when *c(x) ≥ λ*.

## 2. The guarantee (selective risk control)

We want λ such that the **selective error**

    R(λ) = P( ŷ ≠ y | c(x) ≥ λ )

is at most α, with high probability over the calibration data.

Procedure, given *n* labelled calibration decisions, exchangeable with
deployment data:

1. Fix a grid Λ of thresholds *before looking at data* (96 values in [0, 0.999]).
2. For each λ ∈ Λ: let *n_λ* = number of calibration decisions with c ≥ λ and
   *k_λ* = how many of them are wrong. Compute the exact one-sided
   Clopper–Pearson upper bound U(λ).
3. Test the thresholds by **fixed-sequence testing** from K = 4 starting
   thresholds fixed in advance (0.99, 0.95, 0.90, 0.80): each chain walks down
   the grid from its start, certifies λ while U(λ) ≤ α at level 1 − δ/K, and
   stops at its first failure. Choose the smallest certified λ over all chains
   (the most automation). If none is certified, automate nothing.

**Why it is valid.** For a fixed λ, conditional on *n_λ*, the selected
calibration decisions are i.i.d. draws from the deployment distribution
restricted to {c ≥ λ}, so *k_λ ~ Binomial(n_λ, R(λ))* and the Clopper–Pearson
bound is exact. Fixed-sequence testing in an order fixed in advance controls
the family-wise error of one chain at its level δ/K (each hypothesis is only
tested after every earlier one in the chain was rejected, so the first true
null reached is the only way to err, and that test errs with probability
≤ δ/K). A union bound over the K chains gives probability ≤ δ that *any*
certified λ has R(λ) > α, so the chosen one is covered. This does not need the
error to fall monotonically with confidence; monotonicity only makes the
chains go further. This is the Learn-then-Test framework (Angelopoulos, Bates,
Candès, Jordan, Lei 2021) with binomial p-values.

Plain Bonferroni over all |Λ| thresholds is also valid, but tests each one at
δ/|Λ| (here δ/96). On a few hundred calibration decisions it certifies almost
nothing. `calibrate_selective(..., starts=None)` keeps it for comparison. The
chain starts were fixed from the jev-distill and stage-1 predictions, before
any stage-2 result was scored with them.

**Known weakness.** A chain stops at its first failure. If error rises in a
band of high confidence (say 0.93–0.97), every chain that walks through the
band stops there, and the thresholds below it are never tested, even when they
would pass. The rule stays valid but can automate less than Bonferroni would.
`test_conformal.py::test_guarantee_holds_when_error_is_not_monotone_in_confidence`
builds exactly this case: there, fixed-sequence testing certifies a threshold
in about 25% of calibration draws and Bonferroni in about 66%, with violation
rates of about 1% and 0%. Testing every threshold at δ with no correction
violates about 18% of the time (more than δ), so the test can tell a valid rule
from an invalid one. On our models' real predictions error falls with
confidence, and fixed-sequence testing automates more (see the results).

**What it does not need:** calibrated probabilities. A badly calibrated model
still gets a valid guarantee; it just automates less.

**What it does need:** exchangeability. Under distribution shift the promise is
void. `pipeline.shift_test` calibrates in-distribution and deploys on the OOD
split to measure how badly it breaks.

`test_conformal.py::test_selective_guarantee_holds_on_fresh_data` checks the
violation rate empirically (200 trials, fresh test data each time).

## 3. Prediction sets (split conformal, LAC)

Non-conformity *s = 1 − p(y|x)*; *q̂* = the ⌈(n+1)(1−α)⌉/n empirical quantile
on calibration data; set *C(x) = {k : 1 − p_k ≤ q̂}*. Then
P(y ∈ C(x)) ≥ 1 − α (Vovk; Sadinle, Lei, Wasserman 2019).

Automating only singleton sets gives P(automated ∧ wrong) ≤ α as a share of
**all** traffic. That is a different, weaker promise than §2; we report both.

## 4. Model

Cross-encoder: input = `"[kind] option: <option> || question: <q>"` paired with
the state; output = one logit. Softmax over a question's options.

* **Order invariance is exact.** Permuting options permutes logits, so the
  answer can never depend on option order. Listwise models have to learn this
  or average over permutations.
* **Cost.** K encoder passes per K-option question. With 42M non-embedding
  parameters this is still a few ms per option on CPU.
* **Portable.** A standard `AutoModelForSequenceClassification(num_labels=1)`.

## 5. Training objective

    L = CE(p, t) + 0.5·Brier(p, t) + 0.5·RPS(p, t)[ordinal rows] + w·ConfTr

* **CE vs soft targets** (teacher distributions): distillation.
* **Brier**: strictly proper, bounded, punishes confident mistakes.
* **RPS** on `score` questions: respects order (4 vs 5 is a small mistake, 0 vs 5 a big one).
* **ConfTr** (Stutz et al. 2022): split each batch, compute the α-quantile of
  true-class probabilities on one half (the THR threshold), soft set membership
  on the other half, penalise expected set size above 1. Turned on after 30% of
  training so the model first learns sensible probabilities.

Model selection: best validation **NLL** (a proper score), not accuracy.

**Two stages.** Stage 1 trains on jev-distill `openjev_v2` plus the typed-decisions
train carve (384 tokens). Stage 2 starts from the stage-1 baseline and fits the
typed-decisions train carve alone (640 tokens, so no typed-decisions input is
truncated). The two datasets are labelled by different teacher models, and
typed-decisions scores agreement with its own teacher, so a model that keeps
mixing both learns a blend of the two. The stage-2 ConfTr ablation starts from
the same checkpoint with the same data and seed as its baseline. A third run fits
typed-decisions alone from mmBERT-small, to measure what stage 1 adds.

## 6. Calibration order

Stage-2 models (the release candidates) are calibrated on a calibration carve
of typed-decisions `train` (by case; disjoint from the train and validation
carves; `test` is never used), so their guarantee refers to typed-decisions data.
Stage-1 models use that carve plus the jev-distill `calibration` split, and their
guarantee refers to the mixture.

1. Set A: one temperature per question kind (noul / choice / score), fitted by NLL.
2. Set B: the selective threshold of §2 on the temperature-scaled confidences.

Stage-1 models split their calibration pool into two halves by case (A and B).
Stage-2 models use the typed-decisions validation carve as A, so the whole
calibration carve is B (`calibrate_checkpoint(..., temperature_set=...)`); the
validation carve also picked the checkpoint, which is fine because B is never
seen before the threshold is set. Disjoint sets keep the guarantee valid (the
temperatures are fixed before §2 sees B).

## 7. Evaluation protocol

* **Pools**: typed-decisions `test` (headline; 400 cases, 2,000 decisions),
  jev-distill `test_set_30k` (in-distribution, never trained on) and `ood`.
* **typed-decisions official metrics**: accuracy against the gold label,
  KL(gold ‖ prediction), Brier against the soft gold, ECE. Our scorer reproduces
  the benchmark's Uniform reference row (KL 0.444, Brier 0.238); see
  `tests/test_typed_decisions.py`. The benchmark leaves ECE binning open; we use
  15 equal-width bins on top-option confidence.
* **Automation @ α**: 100 random calibration/test splits of each pool, by case
  (the five questions about one case never straddle the split). Report the mean
  automation rate, the realised selective error and the violation rate, and plot
  the realised error of every split (the validity figure).
* **Intervals**: 95% bootstrap intervals resampling whole cases, so correlated
  questions do not make intervals look tighter than they are.
* **Ablation**: identical config with and without ConfTr, same seed (and, in
  stage 2, the same starting checkpoint); the automation difference is paired
  split by split. GPU kernels are not bitwise deterministic: in stage 2 the two
  runs already differ slightly before ConfTr switches on (step-50 training loss
  0.884 vs 0.889 on an A100). Each configuration is a single seed, so read the
  ablation's spread over splits as a lower bound on the uncertainty.
* **Shipped threshold**: the threshold saved with the model, applied unchanged
  to typed-decisions `test`.
* **Shift test**: threshold from `test_set_30k`, deployed on `ood`.
* **Baselines**: Phocinae-Largha-150M-v1 (same mmBERT-small base), called through
  its own engine with each question's instructions and criteria, as in its
  published evaluation; only reported if our harness reproduces its published
  accuracy within one point. Prior and Uniform reference rows as in the benchmark.

## References

* Stutz, Dvijotham, Cemgil, Doucet. *Learning Optimal Conformal Classifiers.* ICLR 2022.
* Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021.
* Geifman, El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017.
* Sadinle, Lei, Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019.
* Marone et al. *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning.* 2025.
