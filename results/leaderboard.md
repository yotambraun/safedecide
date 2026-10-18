### typed-decisions test: official metrics (2,000 decisions)

| Model | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | noul | choice | score |
|---|---|---|---|---|---|---|---|
| SafeDecide (ConfTr) | 0.753 | 0.130 | 0.070 | 0.153 | 0.837 | 0.722 | 0.715 |
| SafeDecide (no ConfTr) | 0.754 | 0.126 | 0.069 | 0.147 | 0.840 | 0.717 | 0.718 |
| SafeDecide, typed-decisions only | 0.745 | 0.126 | 0.067 | 0.146 | 0.837 | 0.720 | 0.694 |
| SafeDecide general (ConfTr) | 0.657 | 0.201 | 0.114 | 0.065 | 0.747 | 0.597 | 0.635 |
| SafeDecide general (no ConfTr) | 0.684 | 0.184 | 0.104 | 0.096 | 0.767 | 0.652 | 0.646 |
| Prior (reference) | 0.487 | 0.328 | 0.181 | 0.037 | 0.622 | 0.452 | 0.414 |
| Uniform (reference) | 0.269 | 0.444 | 0.238 | 0.049 | 0.492 | 0.198 | 0.155 |
| Phocinae-Largha-150M-v1 (test-exposed) | 0.906 | 0.055 | 0.025 | 0.252 | 0.932 | 0.925 | 0.873 |

ECE: 15 equal-width bins on top-option confidence.


### typed_decisions_test: automation at a guaranteed error ≤ 10%

| Model | n | Accuracy | ECE ↓ | NLL ↓ | Automation @ 10% ↑ | Realised error | Violation rate |
|---|---|---|---|---|---|---|---|
| Phocinae-Largha-150M-v1 (test-exposed) | 2000 | 0.906 [0.892, 0.919] | 0.252 [0.239, 0.265] | 0.822 | **93.9%** ± 2.9% | 7.84% | 3% |
| SafeDecide (no ConfTr) | 2000 | 0.754 [0.734, 0.776] | 0.147 [0.130, 0.167] | 0.894 | **33.2%** ± 5.2% | 6.33% | 7% |
| SafeDecide (ConfTr) | 2000 | 0.753 [0.731, 0.775] | 0.153 [0.135, 0.173] | 0.898 | **29.7%** ± 5.2% | 6.27% | 10% |
| SafeDecide, typed-decisions only | 2000 | 0.745 [0.722, 0.766] | 0.146 [0.128, 0.166] | 0.894 | **23.6%** ± 10.6% | 5.46% | 6% |
| SafeDecide general (no ConfTr) | 2000 | 0.684 [0.660, 0.708] | 0.096 [0.075, 0.118] | 0.951 | **2.4%** ± 5.8% | 1.47% | 5% |
| SafeDecide general (ConfTr) | 2000 | 0.657 [0.634, 0.680] | 0.065 [0.050, 0.091] | 0.968 | **1.9%** ± 4.6% | 1.59% | 4% |
| Prior (reference) | 2000 | 0.487 [0.460, 0.515] | 0.037 [0.026, 0.066] | 1.095 | **0.0%** ± 0.0% | 0.00% | 0% |
| Uniform (reference) | 2000 | 0.269 [0.243, 0.293] | 0.049 [0.032, 0.074] | 1.212 | **0.0%** ± 0.0% | 0.00% | 0% |


### jev_test_30k: automation at a guaranteed error ≤ 5%

| Model | n | Accuracy | ECE ↓ | NLL ↓ | Automation @ 5% ↑ | Realised error | Violation rate |
|---|---|---|---|---|---|---|---|
| SafeDecide general (ConfTr) | 2271 | 0.843 [0.827, 0.858] | 0.015 [0.013, 0.031] | 0.356 | **68.7%** ± 2.9% | 3.43% | 7% |
| SafeDecide general (no ConfTr) | 2271 | 0.824 [0.806, 0.841] | 0.019 [0.016, 0.038] | 0.389 | **60.3%** ± 2.7% | 3.41% | 8% |
| SafeDecide (no ConfTr) | 2271 | 0.801 [0.783, 0.817] | 0.032 [0.027, 0.051] | 0.437 | **44.2%** ± 3.4% | 3.14% | 10% |
| SafeDecide (ConfTr) | 2271 | 0.799 [0.781, 0.817] | 0.038 [0.032, 0.054] | 0.441 | **42.7%** ± 2.9% | 3.03% | 8% |
| SafeDecide, typed-decisions only | 2271 | 0.586 [0.563, 0.608] | 0.159 [0.141, 0.181] | 1.005 | **0.0%** ± 0.0% | 0.00% | 0% |


### jev_ood: automation at a guaranteed error ≤ 5%

| Model | n | Accuracy | ECE ↓ | NLL ↓ | Automation @ 5% ↑ | Realised error | Violation rate |
|---|---|---|---|---|---|---|---|
| SafeDecide (ConfTr) | 4000 | 0.753 [0.715, 0.787] | 0.034 [0.023, 0.063] | 0.524 | **3.6%** ± 8.7% | 0.92% | 15% |
| SafeDecide general (no ConfTr) | 4000 | 0.760 [0.719, 0.801] | 0.064 [0.041, 0.098] | 0.556 | **3.2%** ± 8.9% | 0.75% | 12% |
| SafeDecide (no ConfTr) | 4000 | 0.755 [0.725, 0.784] | 0.037 [0.025, 0.062] | 0.534 | **0.8%** ± 4.5% | 0.20% | 3% |
| SafeDecide, typed-decisions only | 4000 | 0.638 [0.581, 0.692] | 0.144 [0.122, 0.194] | 0.923 | **0.0%** ± 0.0% | 0.00% | 0% |
| SafeDecide general (ConfTr) | 4000 | 0.794 [0.777, 0.812] | 0.082 [0.063, 0.103] | 0.647 | **0.0%** ± 0.0% | 0.00% | 0% |
