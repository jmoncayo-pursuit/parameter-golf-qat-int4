# BayesianBackoffCache-TTAdapter

## Naming (read this first)

| Where | Name |
|-------|------|
| **Git branch** | `qat-int4-int6-gps-mlp-tt-adapter` |
| **This note filename / descriptive label** | `BayesianBackoffCache-TTAdapter` |
| **Optional `RUN_ID` / log label shorthand** | `BayesianBackoffCache_TTAdapter` |

Upstream repo URL stays **`jmoncayo-pursuit/parameter-golf-qat-int4`**; the current experiment branch is **`qat-int4-int6-gps-mlp-tt-adapter`**.

## Purpose

Experimental line for **evaluation-time** predictive gains, **not** a new training or export baseline.

**Core question:** Can we improve `val_bpb` at evaluation time using only **already-seen / already-graded** validation tokens, **without** modifying the serialized model artifact?

**Branch role:** this branch is the **combined** experiment. The stable candidate line can remain cache-only. This branch exists to answer the narrower question: does adding `TestTimeAdapter` on top of `BayesianBackoffCache` earn its extra runtime complexity?

## What it tests

### 1. `BayesianBackoffCache`

- Backward-looking **variable-order n-gram** cache.
- During **sliding** validation eval, uses **only already-graded** tokens to build cache statistics and, when thresholds pass, **mix** cache probabilities with model logits.
- **Goal:** lower BPB without changing the saved checkpoint bytes.

### 2. `TestTimeAdapter` (T3)

- Tiny **zero-initialized** eval-only adapter (e.g. bias keyed by hashed bigrams).
- **After** a token is scored, the branch may **update** the adapter from that **already-graded** token (online steps, e.g. `AdamW`), still without writing a new serialized checkpoint.
- **Goal:** measure whether test-time adaptation improves prediction **beyond cache-only** mixing.

## Why `TestTimeAdapter` lives here and not everywhere

- `TestTimeAdapter` is **not** a universal component that every branch should carry.
- Quantization or architecture branches do not need it because it would add a second moving part and muddy attribution.
- The clean ablation is:
  - **candidate branch:** `BayesianBackoffCache` only
  - **this branch:** `BayesianBackoffCache` + `TestTimeAdapter`
- That separation lets us answer the right question: is the adapter itself worth the added runtime and rule-compliance risk?

## Risks to measure

| Risk | What to log |
|------|-------------|
| **Runtime** | Does cache ± TTT stay within the **10-minute** eval budget? Wall-clock per eval. |
| **Rule compliance** | Does this still match the README’s “only already evaluated tokens” intent? (Organizer confirmation if serious.) |
| **Incremental value** | Does the adapter help **beyond** cache-only? Needs a controlled ablation. |

## Evidence status

- **Not paper-backed** for this line; feasibility is **code- and rule-grounded**. Do not cite unrelated arXiv IDs as motivation.
- Prefer **committed** logs: `val_bpb`, wall-clock, and config hash for each run.

## Next concrete ablation

On **target-like hardware**:

1. **`qat-int4-int6-gps-mlp` (candidate):** cache-only path — `eval_val_sliding_cached()` with **`BayesianBackoffCache`** and **no `TestTimeAdapter`**.
2. **`qat-int4-int6-gps-mlp-tt-adapter`:** same checkpoint, same token slice, **cache + `TestTimeAdapter`** enabled.

Log **both** `val_bpb` and **wall-clock** so the line is judged on evidence, not theory.

## Related files (this repository)

- `train_gpt.py`: `BayesianBackoffCache` and `eval_val_sliding_cached()` on **`qat-int4-int6-gps-mlp`**. Branch **`qat-int4-int6-gps-mlp-tt-adapter`** adds `TestTimeAdapter` and T3 updates in that eval loop.
- `run_BayesianBackoffCache_TTAdapter.sh`: optional H100 entrypoint for the **`qat-int4-int6-gps-mlp-tt-adapter`** branch; compare with `run_qat_int4_int6_gps_mlp_baseline.sh` on **`qat-int4-int6-gps-mlp`** for cache-only.

## Legacy naming

Any **`frontier-eval-adaptation`** name is historical only.
