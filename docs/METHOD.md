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

Procedure, given *n* labelled calibration units exchangeable with the units
it will be applied to. A unit is one decision by default; with clustered data it
can be one case (`unit="case"` or `"one-per-case"`, below):

1. Fix a grid Λ of thresholds *before looking at data* (96 values in [0, 0.999]).
2. For each λ ∈ Λ: let *n_λ* = number of calibration decisions with c ≥ λ and
   *k_λ* = how many of them are wrong. Compute the exact one-sided
   Clopper–Pearson upper bound U(λ).
3. Test the thresholds by **fixed-sequence testing** from K = 4 starting
   thresholds fixed in advance (0.99, 0.95, 0.90, 0.80): each chain walks down
   the grid from its start, certifies λ while U(λ) ≤ α at level 1 − δ/K, and
   stops at its first failure. Choose the smallest certified λ over all chains
   (the most automation). If none is certified, automate nothing.

**Why it is valid.** Assume the calibration units are i.i.d. draws from the
distribution deployment units come from (exchangeability is what the argument
uses). For a fixed λ, conditional on *n_λ*, the selected calibration units are
then i.i.d. draws from that distribution restricted to {c ≥ λ}, so
*k_λ ~ Binomial(n_λ, R(λ))* and the Clopper–Pearson bound is exact. Fixed-sequence testing in an order fixed in advance controls
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

**What it does need:** exchangeability of the units it certifies with the units
it is applied to. Under distribution shift the promise is void.
`pipeline.shift_test` calibrates in-distribution and deploys on the OOD split to
measure how badly it breaks.

**Clustered calibration data.** Calibration decisions are rarely independent.
typed-decisions asks five questions about each case, and a file of your own
labelled decisions usually asks several questions about one ticket. Questions
about one case share its state, and their errors are correlated: a confusing
case tends to be answered wrongly more than once. Counting each decision as a
unit then assumes more independence than the data has. *k_λ* is a sum of
positively correlated indicators, its variance is larger than the binomial's,
and the Clopper–Pearson bound, which reads *n_λ* as *n_λ* independent draws, can
be too low: the 1 − δ coverage is no longer guaranteed. The error estimate itself
stays unbiased; what is overstated is the certainty about it. What is
exchangeable in such data is whole cases.

Two certificates rely only on whole cases being exchangeable (`safedecide calibrate
--unit case` or `--unit one-per-case`; `unit=` in `certify_rules`,
`calibrate_checkpoint`, `calibrate_adapter` and `automation_at_alpha`).
`calibrate` uses the `case` field of each line when there is one, and the state
otherwise. When cases repeat and the unit is `decision`, it warns
(`ClusteredCalibrationWarning`). Both promise the same thing: the error of a
uniformly random question of a new case, among automated ones. That is the
per-decision error when every case asks the same number of questions, as in
typed-decisions; when the number varies, it weights cases equally rather than
decisions.

**One decision per case** (`one-per-case`, or `conformal.one_per_case` for your own
confidences) keeps one decision of each calibration case, drawn uniformly at
random with a fixed seed, and certifies on those with the procedure above. If
cases are exchangeable these are i.i.d. draws, so the argument holds exactly. It
discards every other question, and the outcome depends on the draw.

**Every question, one unit per case** (`case`, `conformal.calibrate_selective_by_case`).
For a threshold λ and case *i* with *m_i* questions, let *S_i* be how many are
automated and *W_i* how many of those are wrong, and

    x_i = (W_i − α S_i) / m_i  ∈ [−α, 1 − α].

The promised error is at most α exactly when E[x] ≤ 0. For a fixed bet *b* in
[0, 1/(1 − α)), every factor 1 − b·x_i is positive, and if cases are independent,
E[∏ᵢ (1 − b·x_i)] = ∏ᵢ (1 − b·E[x]) ≤ 1 when E[x] > 0. That holds however the
questions of one case depend on each other. The average of this product over 200
fixed bets is therefore an e-value against "the error exceeds α". A threshold is
certified when it reaches K/δ, which by Markov's inequality errs with probability at
most δ/K, inside the same fixed-sequence chains. The bets are fixed in advance, as a
midpoint quadrature of a Beta(1, ½) prior on *b*(1 − α), which favours the largest
admissible bets. No data chose them, and the order of the cases does not matter.
This is testing by betting (Waudby-Smith and Ramdas, 2024) applied to a bounded
per-case quantity; `calibration_error_ucb` is the smallest α′ at which the test
still rejects.

What it costs is caution, not data. Because it assumes nothing about how the
questions of one case are related, it must allow for all of them failing together,
which caps the bet at 1/(1 − α): a case contributes at most what one decision would.
On the saved per-decision test predictions of SafeDecide-140M v0.1 (typed-decisions
`test`, 2,000 decisions in 400 cases), over the same 100 case splits as the results
(half the cases calibrate, the other half test), with automation on the test half:

| Budget | All decisions as units | Every question, one unit per case | One decision per case |
|---|---|---|---|
| 5% | 4.7% automated, certified in 32 of 100 splits | 0.0%, certified in none | 0.0%, certified in none |
| 10% | 32.4% ± 5.5, certified in 100 of 100 | 0.0%, certified in none | 3.9% ± 10.1, certified in 14 |
| 20% | 70.3% ± 5.5, certified in 100 of 100 | 63.4% ± 5.9, certified in 100 | 43.8% ± 23.3, certified in 88 |

The test half went over budget in at most 2 of 100 splits for both case-level
certificates (the all-decisions rule: 9 at 10%, 10 at 20%). On the ConfTr variant's
predictions the case-level certificate gives 61.0% at 20% (one per case: 42.0%).
(The all-decisions row recomputes the README's 33.2% from these files and gets
32.4%.) At 20% the case-level certificate keeps most of the all-decisions
automation; at 10%, 200 calibration cases are not enough evidence for a promise
that must hold even if every question of a case fails at once. With more cases it
certifies lower budgets too.

In the simulation in
`test_conformal.py::test_by_case_stays_valid_on_clustered_data_where_all_rows_do_not`,
where every question of a case shares its outcome, the all-decisions rule exceeds a
20% budget in about 16% of calibration draws (δ = 10%), and the case-level
certificate in about 0.3% while certifying in about 92%. At 10%, the all-decisions
rule exceeds its budget in about 17% of draws and one decision per case in under 1%
(`test_one_per_case_stays_valid_on_clustered_data_where_all_rows_do_not`).

The shipped thresholds and every *Automation @ α* figure in the results count
decisions as units (the shipped ones: 600 decisions in 120 cases), so they rest on
decision-level exchangeability, which clustering violates. They are left as
published. On typed-decisions `test`, the share of the 100 splits whose test half
went over budget was 0.09 at 10% and 0.10 at 20% for v0.1 (all-decisions rule), at
most δ. That check also runs on case-correlated test halves, so it is reassuring,
not a proof.

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
restores the two-halves split. With `--unit case` or `--unit one-per-case`, the
threshold is certified with the cases of set B as the units (§2).

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
* **A 400M model**: the winning stage-2 recipe, fixed in advance, straight from
  Ettin-encoder-400m with no stage 1 and no cross-validation of its own. It trains once on
  90% of the 1,080 cases, with early stopping and temperatures on the other 10%, and is
  certified on the same calibration carve and sealed with the 140M model.

### What v0.2 found

Out-of-fold accuracy on 5,400 decisions, paired against v0.1's schedule (95% intervals
over cases):

| Recipe | Accuracy | vs `r0-old-schedule` |
|---|---|---|
| `r0-old-schedule` | 0.658 | – |
| `r1-long` | 0.664 | +0.005 [−0.008, +0.018] |
| `r1-long-s1x2` | 0.737 | +0.079 [+0.064, +0.093] |
| `r2-factors` (chosen) | 0.748 | +0.089 [+0.074, +0.104] |
| `r2-factors` + per-question bias | 0.743 | +0.084 [+0.069, +0.099] |

* A longer stage 1 is what helped. A longer stage 2 alone did not, and factor questions
  added about one point on top (+0.011 [−0.001, +0.022] over `r1-long-s1x2`, short of clear).
* Keeping typed-decisions out of stage 1 cost more than expected. On v0.1's own
  600-decision validation carve, `r0-old-schedule` is 9.8 points below v0.1, whose stage 1
  saw the typed-decisions train carve four times. The chosen recipe only ties v0.1 there
  (+0.002 [−0.037, +0.038]). The leak-free stage 1 made the comparison clean but removed
  v0.1's strongest ingredient.
* On `test`, scored once after the seal: the 400M model reached 0.787 (+0.033
  [+0.014, +0.052] over v0.1), the 140M v0.2 0.748 (−0.006 [−0.026, +0.011]). The gains
  measured on the calibration carve before sealing (+0.048 and +0.018) were both
  larger. That carve also sets the thresholds and was the go/no-go check, so a smaller
  gain on `test` is what selection on it predicts.
* Precision moves a single score by up to 0.7 points but not the comparison. The test
  pass ran with bf16 autocast on an A100, while the thresholds were certified in full
  precision. In full precision on CPU the 400M scores 0.7875, the 140M v0.2 0.755 and
  v0.1 0.7545; paired against v0.1 in full precision, the 400M is +0.033
  [+0.014, +0.052] and the 140M v0.2 +0.0005 [−0.021, +0.019]. `LocalAdapter` now
  always scores in full precision.
* Accuracy conventions differ on the 35 test decisions whose gold probabilities tie.
  Against the gold `label`, the 400M scores 0.787 (used throughout). Counting any tied
  option as correct gives 0.794, and taking the first tied option, as the benchmark's
  `bench/score_with_jobs.py` does, gives 0.784.

## References

* Stutz, Dvijotham, Cemgil, Doucet. *Learning Optimal Conformal Classifiers.* ICLR 2022.
* Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control.* 2021.
* Geifman, El-Yaniv. *Selective Classification for Deep Neural Networks.* NeurIPS 2017.
* Sadinle, Lei, Wasserman. *Least Ambiguous Set-Valued Classifiers with Bounded Error Levels.* JASA 2019.
* Marone et al. *mmBERT: A Modern Multilingual Encoder with Annealed Language Learning.* 2025.
* Weller et al. *Seq vs Seq: An Open Suite of Paired Encoders and Decoders.* 2025 (Ettin).
