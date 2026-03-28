# Branch: `qat-int4-int6-gps-mlp-tt-adapter`

**Repository:** same fork as the candidate line — [`parameter-golf-qat-int4`](https://github.com/jmoncayo-pursuit/parameter-golf-qat-int4) (not a separate product repo).

## Purpose

Experimental line for **evaluation-time** predictive gains, **not** a new training or export baseline.

**Core question:** Can we improve `val_bpb` at evaluation time using only **already-seen / already-graded** validation tokens, **without** modifying the serialized model artifact?

**Branch role:** **combined** experiment on top of **`qat-int4-int6-gps-mlp`**. The candidate branch can stay cache-only; this branch tests whether **`TestTimeAdapter`** on top of **`BayesianBackoffCache`** is worth the extra eval cost.

## What it tests

### 1. `BayesianBackoffCache`

- Backward-looking **variable-order n-gram** cache.
- During sliding validation eval, uses **only already-graded** tokens to build cache statistics and, when thresholds pass, **mix** cache probabilities with model logits.
- **Goal:** lower BPB without changing the saved checkpoint bytes.

### 2. `TestTimeAdapter` (T3)

- Tiny **zero-initialized** eval-only adapter (e.g. bias keyed by hashed bigrams).
- **After** a token is scored, the branch may **update** the adapter from that **already-graded** token (online steps, e.g. `AdamW`), still without writing a new serialized checkpoint.
- **Goal:** measure whether test-time adaptation improves prediction **beyond cache-only** mixing.

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
- `run_qat_int4_int6_gps_mlp_tt_adapter.sh`: H100 entrypoint on this branch; compare with `run_qat_int4_int6_gps_mlp_baseline.sh` on **`qat-int4-int6-gps-mlp`** for cache-only.

## Naming

- **Python types** stay `BayesianBackoffCache` and `TestTimeAdapter` (implementation names).
- **Git branch** and **docs** use the `qat-int4-int6-gps-mlp*` prefix for consistency with the candidate stack.
