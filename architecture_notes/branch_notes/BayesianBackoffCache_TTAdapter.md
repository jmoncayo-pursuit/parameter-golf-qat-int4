# Branch / repo: `BayesianBackoffCache_TTAdapter`

## Purpose

Experimental line for **evaluation-time** predictive gains, **not** a new training or export baseline.

**Core question:** Can we improve `val_bpb` at evaluation time using only **already-seen / already-graded** validation tokens, **without** modifying the serialized model artifact?

## What it tests

### 1. `BayesianBackoffCache`

- Backward-looking **variable-order n-gram** cache.
- During **sliding** validation eval, uses **only already-graded** tokens to build cache statistics and, when thresholds pass, **mix** cache probabilities with model logits.
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

1. **`int6-gps-int4-mlp` (candidate):** cache-only path — `eval_val_sliding_cached()` with **`BayesianBackoffCache`**, adapter **disabled** if gated by flag.
2. **`BayesianBackoffCache_TTAdapter`:** same checkpoint, **cache + `TestTimeAdapter`** enabled.

Log **both** `val_bpb` and **wall-clock** so the line is judged on evidence, not theory.

## Related files (this repository)

- `train_gpt.py`: `BayesianBackoffCache` and `eval_val_sliding_cached()` on **`int6-gps-int4-mlp`**. Branch `BayesianBackoffCache_TTAdapter` adds `TestTimeAdapter` and T3 updates in that eval loop.
- `run_bayesian_backoff_cache_tt_adapter.sh`: optional H100 entrypoint on this branch; compare with `run_baseline.sh` on **`int6-gps-int4-mlp`** for cache-only.

## GitHub

Rename the remote repository from any legacy name (e.g. `frontier-eval-adaptation`) to **`BayesianBackoffCache_TTAdapter`** so URLs match this note.
