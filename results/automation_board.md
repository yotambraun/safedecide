| Model | Training | Accuracy | Automation @ 5% | @ 10% | @ 20% | @ 10%, Bonferroni |
|---|---|---:|---:|---:|---:|---:|
| Phocinae-Largha-150M-v1 (test-exposed) | fitted, test-exposed | 0.913 | 72.6% | 97.2% ± 2.7% | 100.0% | 91.8% |
| vllm-sr/d3 | general, zero-shot | 0.749 | 0.6% | 35.9% ± 6.2% | 73.9% | 27.7% |
| perplexity-ai/pplx-decider-v1.1-27b | general, zero-shot | 0.753 | 2.1% | 35.3% ± 4.6% | 70.9% | 26.1% |
| SafeDecide (no ConfTr) | ours, fitted on train | 0.760 | 4.4% | 33.8% ± 5.4% | 73.1% | 24.6% |
| autotrust/JEV-27B-VL | general, zero-shot | 0.688 | 20.4% | 33.0% ± 2.1% | 52.5% | 30.2% |
| mrm8488/embeddinggemma-2-ft-typed-decision | fitted on train | 0.770 | 1.0% | 31.9% ± 6.9% | 78.4% | 22.7% |
| vllm-sr/d3-flash | general, zero-shot | 0.717 | 1.6% | 30.2% ± 3.6% | 62.6% | 23.6% |
| SafeDecide (ConfTr) | ours, fitted on train | 0.760 | 4.0% | 30.1% ± 5.4% | 72.6% | 22.5% |
| Cloudflare/clef | general, zero-shot | 0.728 | 16.5% | 28.8% ± 4.1% | 63.9% | 22.7% |
| openjev/openjev | general, zero-shot | 0.728 | 5.5% | 26.7% ± 5.0% | 58.5% | 20.4% |
| SafeDecide, typed-decisions only | ours, fitted on train | 0.753 | 0.8% | 25.0% ± 10.9% | 72.9% | 15.3% |
| vllm-sr/d3-mini | general, zero-shot | 0.743 | 3.5% | 22.0% ± 3.3% | 63.1% | 16.9% |
| Cloudflare/clef-flash | general, zero-shot | 0.710 | 0.1% | 17.4% ± 10.6% | 59.2% | 6.2% |
| Cloudflare/clef-omni | general, zero-shot | 0.673 | 0.2% | 11.0% ± 8.4% | 51.5% | 2.8% |
| autotrust/JEV-35B | general, zero-shot | 0.700 | 0.3% | 10.5% ± 10.8% | 55.8% | 2.1% |
| SafeDecide general (no ConfTr) | ours, general | 0.692 | 0.0% | 2.4% ± 5.8% | 42.3% | 0.7% |
| SafeDecide general (ConfTr) | ours, general | 0.666 | 0.0% | 1.9% ± 4.6% | 33.5% | 0.2% |
| LiquidAI/d1-3B | general, zero-shot | 0.656 | 0.0% | 0.3% ± 1.5% | 30.5% | 0.1% |
| autotrust/GEV-26B-Decide | general, zero-shot | 0.522 | 0.0% | 0.1% ± 0.9% | 6.3% | 0.0% |
| vllm-sr/d3-lite | general, zero-shot | 0.530 | 0.0% | 0.1% ± 0.6% | 9.4% | 0.0% |
| nandakishorm/vega-08b-public-intents | fitted on train | 0.663 | 0.0% | 0.0% ± 0.0% | 24.4% | 0.1% |
| vllm-sr/d3-nano | general, zero-shot | 0.629 | 0.0% | 0.0% ± 0.0% | 3.4% | 0.0% |
| Prior (reference) | reference | 0.495 | 0.0% | 0.0% ± 0.0% | 1.3% | 0.0% |
| autotrust/JEV-27B | general, zero-shot | 0.495 | 0.0% | 0.0% ± 0.0% | 0.5% | 0.0% |
| vllm-sr/d3-edge | general, zero-shot | 0.425 | 0.0% | 0.0% ± 0.0% | 0.5% | 0.0% |
| doofz/THX-01 | general, zero-shot | 0.377 | 0.0% | 0.0% ± 0.0% | 0.0% | 0.0% |
