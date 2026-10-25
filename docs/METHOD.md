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

**Error levels.** Every checkpoint is certified at α = 5%, 10% and 20%, each
by the procedure above, and the user picks the level that matches their error
budget. Each level is its own guarantee; picking the level that happens to
automate most, after seeing the results, is not covered. The typed-decisions
release defaults to 10%. Its gold labels are the mean of three teacher samples
(teacher self-agreement 0.735), so even confident answers carry label noise, and
600 calibration decisions rarely certify 5%. On Hugging Face's published
results, most models automate under 6% at that level.

**What it does not need:** calibrated probabilities. A badly calibrated model
still gets a valid guarantee; it just automates less.

**What it does not give:** a guarantee per task. The bound is on the error
over the whole mix the calibration set was drawn from. On jev-distill `ood`,
at one certified threshold (0.95, ConfTr stage-2 model) the pooled error was
4.3%, but 11.6% on `snake-v1` and 1.4% on `workflow-controls-v1`. If one task
matters on its own, calibrate on that task.

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
* **Cost.** K encoder passes per K-option question, run as one batch. Measured
  on one CPU thread of a Colab VM, a short 3-option decision takes about 83 ms
  in fp32 ONNX (p50; the model card's Speed table). Long inputs cost more:
  attention grows with the 640-token context.
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

Model selection: best validation **NLL** (a proper score), not accuracy. Since v0.2,
training stops early once the validation NLL has not improved for 5 evaluations
(`patience`), and the recipe itself is chosen by cross-validation (§8).

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

On your own labelled decisions (`safedecide calibrate --data`), the checkpoint's
saved temperatures are kept by default and the whole file is set B. The
temperatures were fitted before that file was seen, so the guarantee is
unaffected, and none of a small file is spent on them. `--refit-temperatures`
restores the two-halves split.

The release is the stage-2 model with the lowest validation NLL. Choosing it by
automation on B would reuse the data behind its guarantee.

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

## 8. Choosing a recipe without test (v0.2)

v0.1 picked its checkpoint on a 600-decision validation carve, where accuracy
moves by ±1.5 points between evaluations, and every v0.1 run was still improving
at its last step. v0.2 changes are chosen in `notebooks/03_improve_gpu.ipynb`:

* **Nested 5-fold cross-validation** over the 1,080 typed-decisions `train` cases
  outside the calibration carve. In fold *i*, a 10% inner carve of the other folds
  picks the checkpoint (early stopping on NLL) and fits the temperatures; fold *i*
  is only predicted. The 5,400 out-of-fold decisions score every candidate, and
  differences are paired, with 95% intervals from resampling whole cases.
* **Stage 1 without typed-decisions**, so no fold leaks into the starting point.
* **Candidates**: v0.1's stage-2 schedule; up to 10 epochs with early stopping;
  the same on a stage 1 trained twice as long; the better of those plus *factor
  questions*; and a per-question bias (fitted on four folds, applied to the fifth).
  The highest out-of-fold accuracy wins; the bias is kept only if it raises
  accuracy without raising KL.
* **Factor questions.** Each `train` case records the latent facts it was
  generated from. During training only, each usable fact (yes/no, a category of
  at most 15 short values, or an integer with at most 10 levels: 31 across the four
  workflows) becomes one more question about the same state with a one-hot target.
  They are never model input, and test-split factors are never loaded.
* **Refit**: the winner is trained once on all 1,080 cases with the same LR
  schedule, stopped at the folds' median best epoch. Temperatures are the folds'
  mean, and the threshold is certified on the untouched calibration carve, as in §6.
* **Seal**: before `test` is loaded, a recipe file records the code commit, every
  out-of-fold score, the choice and the SHA-256 of each final model's weights and
  configuration. Its hash is posted publicly, and the scoring pass refuses to run
  if any of it has changed.

## References

* Stutz, Dvijotham, Cemgil, Doucet. *Learning Optimal Conformal Classifiers.* ICLR 2022.
* Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021.
* Geifman, El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017.
* Sadinle, Lei, Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019.
* Marone et al. *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning.* 2025.
