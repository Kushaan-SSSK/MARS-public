# MARS benchmarks

These results are research benchmarks from the MARS evaluation dataset. They describe agreement against the study reference labels and runtime behavior in the stated tests. They are not a guarantee of performance on a new acquisition system, animal, or experimental protocol.

Two evaluation sets appear below. The **full labeled set** is every clean recording that carried usable reference labels and is used for the baseline-model agreement and per-state analysis. The **held-out set** is a subject-disjoint subset used for the fair model comparison, so a model is never evaluated on a subject it was trained on. Metrics differ slightly between the two because the sets and the trained models differ.

## Evaluation dataset

- 112 clean recordings were assembled; 102 carried usable reference labels.
- Labels use 2.5-second epochs exported from the reference scorer.
- 4.26 million strict scored epochs, representing 2,961 hours of strict labeled time.
- Transitional and unclassified epochs are excluded before strict metrics are computed. Randomized labels assign bounded neighbor states.
- The held-out comparison contains 24 recordings and 1,005,992 scored epochs.

### Reference-label composition

Across the 102 labeled recordings (total labeled time 3,058.5 hours):

| State | Epochs | Share |
| --- | ---: | ---: |
| Wake | 1,954,831 | 44.4% |
| NREM | 2,078,352 | 47.2% |
| REM | 230,622 | 5.2% |
| Transitional / unclassified | 140,431 | 3.2% |

REM is by far the smallest class, which makes it the hardest state to predict and the main driver of per-state metric spread.

## Held-out model comparison

Strict held-out metrics on 24 subject-disjoint recordings (1,005,992 scored epochs). The MARS default offline model led every metric.

| Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 | Cohen's kappa |
| --- | ---: | ---: | ---: | ---: | ---: |
| MARS default offline | 0.969 | 0.961 | 0.914 | 0.971 | 0.945 |
| MARS real-time (E2.5W9) | 0.962 | 0.950 | 0.904 | 0.964 | 0.931 |
| AccuSleePy | 0.963 | 0.930 | 0.930 | 0.963 | 0.932 |
| IntelliSleepScorer | 0.881 | 0.785 | 0.795 | 0.880 | 0.782 |
| REST | 0.775 | 0.714 | 0.734 | 0.767 | 0.586 |

REM was the smallest class in the evaluation set and the most difficult state near class boundaries. Interpret per-state conclusions with that class imbalance in mind.

## Full labeled-set baseline agreement

Overall agreement of each third-party scoring tool against the reference labels on the full labeled set (strict, 2.5-second epochs). These numbers use the full set, so they differ from the held-out table above.

| Model | Accuracy | Macro F1 | Cohen's kappa |
| --- | ---: | ---: | ---: |
| AccuSleePy | 0.950 | 0.904 | 0.908 |
| IntelliSleepScorer | 0.868 | 0.769 | 0.758 |
| REST | 0.707 | 0.618 | 0.461 |

AccuSleePy was the strongest off-the-shelf baseline and the basis for the MARS default and real-time models. IntelliSleepScorer was clearly above REST. REST agreement was too low to serve as a real-time baseline.

### Per-state F1 (full labeled set, strict)

| State | REST | AccuSleePy | IntelliSleepScorer |
| --- | ---: | ---: | ---: |
| Wake | 0.76 | 0.96 | 0.89 |
| NREM | 0.65 | 0.96 | 0.87 |
| REM | 0.45 | 0.80 | 0.54 |

Wake and NREM are comparatively easy for all tools. REM has fewer examples and sits closer to state boundaries, so every tool drops there.

### Confusion matrices (row-normalized, strict)

Each row is a true reference state; each column is the predicted state. Rows sum to 1.

AccuSleePy:

| True \ Pred | Wake | NREM | REM |
| --- | ---: | ---: | ---: |
| Wake | 1.00 | 0.00 | 0.00 |
| NREM | 0.06 | 0.93 | 0.01 |
| REM | 0.18 | 0.09 | 0.73 |

IntelliSleepScorer:

| True \ Pred | Wake | NREM | REM |
| --- | ---: | ---: | ---: |
| Wake | 0.96 | 0.03 | 0.01 |
| NREM | 0.15 | 0.83 | 0.02 |
| REM | 0.23 | 0.31 | 0.45 |

REST:

| True \ Pred | Wake | NREM | REM |
| --- | ---: | ---: | ---: |
| Wake | 0.99 | 0.01 | 0.00 |
| NREM | 0.51 | 0.48 | 0.01 |
| REM | 0.63 | 0.06 | 0.31 |

REST misclassifies about half of NREM as Wake and most REM as Wake, which is why its kappa is the lowest.

## Real-time model timing evaluation

| Model | Accuracy | Balanced accuracy | Macro F1 | Missed deadlines | p95 inference |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2.0W3 | 95.65% | 95.68% | 95.70% | 0 / 300+ epochs | 0.788 ms |
| E2.0W1 | 92.33% | 92.44% | 92.48% | 0 / 400+ epochs | 1.022 ms |
| E2.5W1 | 93.75% | 93.86% | 93.86% | 0 / 300+ epochs | 1.479 ms |
| E2.5W9 | 94.83% | 94.85% | 94.84% | 0 / 200+ epochs | 11.566 ms |

The four real-time models were trained from the AccuSleePy pipeline. E2.0W3 is the default real-time profile: best accuracy with the lowest p95 inference time and no missed deadlines. The real-time results used causal feature extraction and the model-specific timing settings used in the evaluation. They should be revalidated after changing hardware, signal scaling, channels, or model configuration.

### Runtime and packaging

- Inference backend: CPU, ONNX Runtime 1.18.1.
- ONNX model size: 27-35 KB per model.
- Default real-time model: E2.0W3.
- Processing sample rate after scaling: 1 kHz.
- The runtime handles model metadata, calibration, feature extraction, stimulation rules, and logging.

## Live dry-run evidence

The included [single-subject example](../examples/realtime_scoring_single_subject/README.md) is a 6 minute 34 second live-acquisition dry-run: 195 scored epochs (180 Wake, 15 REM), zero missed deadlines, and a p95 latency of 1.18 ms. It validated live acquisition, packaged calibration loading, ONNX model inference, epoch timing, stim-decision logging, and clean run-summary generation. It did not enable physical output; hardware stimulation requires a separate bench and animal-facing validation, where the animal must reach qualifying NREM to confirm an automatic TTL trigger.

## Multi-subject stress testing

| Active slots | Epochs per slot | CPU average | Memory growth | p95 | p99 | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 900 | 3.01% | -5.4 MB | 0.86 ms | 1.06 ms | 0 |
| 2 | 900 | 3.29% | -2.2 MB | 1.24 ms | 1.48 ms | 0 |
| 4 | 212 | 3.92% | 10.5 MB | 1.59 ms | 1.83 ms | 0 |
| 8 | 900 | 3.27% | 2.8 MB | 1.50 ms | 1.67 ms | 0 |
| 12 | 900 | 3.34% | 13.2 MB | 1.39 ms | 1.46 ms | 0 |
| 16 | 900 | 4.07% | 8.9 MB | 1.99 ms | 2.17 ms | 0 |
| 20 | 900 | 3.93% | 1.8 MB | 1.78 ms | 2.43 ms | 0 |

The four-slot row was explicitly marked for rerun in the source evaluation because it contains fewer epochs. The 20-slot stress test ran for 31.7 minutes, scored 952 epochs per slot, and reported a maximum per-slot p95 latency of 1.78 ms with 0 misses (CPU average 3.93%, memory growth 1.8 MB). Slots A1-A8 map to hardware TTL lines 1-8; A9-A20 log scores and default to TTL 0 until additional TTL wiring is added.
