# Branch: `noisy-qat-bayesian`

**Repository:** same fork as the candidate line — [`parameter-golf-qat-int4`](https://github.com/jmoncayo-pursuit/parameter-golf-qat-int4).

## Purpose

This branch preserves the **original intended training-side experiment** that later drifted into a different eval-time line.

**Core question:** can a more quantization-robust training path, built around **noisy QAT + a Bayesian-inspired quantization idea**, improve the existing mixed-precision export stack **before** relying on eval-time cache or test-time adaptation tricks?

## Why this branch exists

The earlier idea was a **two-part hypothesis**:

1. **Training-side robustness:** make the model less fragile to low-bit export by treating quantization as a source of noise / regularization, potentially with a Bayesian-inspired or uncertainty-guided component.
2. **Eval-time adaptation:** use already-graded tokens to recover additional predictive accuracy at evaluation time.

Only the **second** part materially landed in code on **`qat-int4-int6-gps-mlp-tt-adapter`**. That branch now honestly represents the eval-time line: **`BayesianBackoffCache`** + **`TestTimeAdapter`**.

This branch exists so the **first** part has its own clean lineage and does not look like it was silently abandoned or mislabeled.

See also: [qat-int4-int6-gps-mlp-tt-adapter.md](./qat-int4-int6-gps-mlp-tt-adapter.md)

## Inspiration

- **Competition pressure:** the leaderboard snapshot in the repository [README](../../README.md#leaderboard) already shows the field leaning toward mixed-precision QAT, sliding evaluation, and even test-time compute.
- **Quantization-as-regularization / noise:** [QReg: On Regularization Effects of Quantization](https://arxiv.org/abs/2206.12372)
- **Probabilistic / uncertainty-guided quantization:** [Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantization](https://arxiv.org/abs/2309.13575)
- **Backward-looking LM interpolation ideas:** [Bayesian Language Model Interpolation for Mobile Speech Input](https://research.google/pubs/pub37567)

The **combination idea** here is the important part: use a training-side robustness intervention together with a separate eval-time adaptation line, rather than pretending either literature alone already defines this exact experiment.

## Intended method

- Start from the current **`qat-int4-int6-gps-mlp`** candidate stack.
- Keep the model architecture and mixed-precision export path recognizable.
- Add a **training-side** intervention first, not an eval-time one.
- Candidate directions for this branch:
  - warmdown-only noisy fake quantization on the aggressive Int4 MLP path
  - a Bayesian-inspired / uncertainty-guided alternative to plain max-abs scaling for fragile blocks
  - bin-center or boundary-avoidance regularization if needed to stabilize low-bit export

## What is implemented now

- The inherited candidate stack from **`qat-int4-int6-gps-mlp`** is present.
- **No branch-specific noisy-QAT or probabilistic-scale code is implemented yet.**
- Right now this branch is a **clean research lineage marker** plus explanation, so the original training-side idea has a place to live before code lands.

## What this branch is not

- **Not** the eval-time cache + adapter experiment. That is **`qat-int4-int6-gps-mlp-tt-adapter`**.
- **Not** a claim that the current code implements a formal Bayesian method.
- **Not** a TurboQuant branch.
- **Not** a measured win yet.

## Next concrete step

Implement the smallest honest training-side intervention first:

1. warmdown-only noisy fake quantization on Int4 MLP weights
2. deterministic fallback flag for clean A/B comparison
3. one branch-specific note documenting exactly what changed and what still remains only an intention
