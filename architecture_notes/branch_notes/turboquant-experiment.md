# Branch: `turboquant-experiment`

Inspiration and feasibility notes live here; Runpod and other short GPU staging runs carry **staged train** evidence against `train_gpt.py` on this branch.

## Purpose
- TurboQuant-style rotation and centroid quantization as a **hypothesis** for Parameter Golf static-weight PTQ (see [TurboQuant paper](https://arxiv.org/abs/2504.19874); weight PTQ is a deliberate transfer, not a claim the paper covers it).
- This branch keeps the **write-up** and `turboquant_mse_probe.py` feasibility tooling; integration into `train_gpt.py` export paths remains staged behind artifact / `val_bpb` checks.

## Verified Now
- The branch contains `turboquant_mse_probe.py`.
- The probe script implements measurement logic for:
  - baseline blocked Int4 quantization,
  - rotation + centroid quantization (`TQ-Centroid`),
  - rotation + baseline block-scaling (`TQ-Hybrid`).
- No Stage 1 repo integration has been implemented in `train_gpt.py` on this branch. The quantize/dequantize/export changes described below remain planned, not coded.

## Feasibility Summary
- **The source paper does not prove this use case.** The memo's core finding is that TurboQuant is demonstrated on KV cache vectors and ANN embeddings, not on static model-weight compression. Weight PTQ is a hypothesis transfer.
- **Stage 1 is a serialization-path-only experiment, not a runtime rewrite.** The memo's recommended first implementation would touch `mixed_quantize_int6()`, `dequantize_mixed_int6()`, and the serialized payload so that MLP weights can be rotated, centroid-quantized, dequantized, and inverse-rotated before `load_state_dict`.
- **Stage 2 is centroid refinement, not a separate architecture idea.** If Stage 1 earns its keep, the next step is better scalar quantization / learned centroids inside the same serialization boundary.
- **Stage 3 is likely budget-negative.** The memo's feasibility analysis says QJL residual correction is probably a bad fit here because the extra residual storage is likely to cost too much artifact budget for static-weight PTQ.
- **Stage 4 is explicitly not recommended early.** Simulating TurboQuant inside training/QAT would expand scope into forward/training dynamics without paper support for this use case.
- **The rotation overhead is material but not obviously fatal.** The memo derived a shared 512x512 float16 rotation matrix as `524,288` bytes uncompressed, with compressed size expected to stay around the ~0.5MB range. That is significant but not automatically disqualifying under a 16MB budget.

## Exact Insertion Points
- Add a new rotational quantizer helper near the existing blocked quantization helpers.
- Modify `mixed_quantize_int6()` to add an MLP branch for rotational quantization.
- Modify `dequantize_mixed_int6()` to add the matching inverse-rotation branch.
- Extend the quantized `torch.save({"w": ..., "m": ...})` payload to carry the shared rotation matrix and any centroid metadata.
- Leave `GPT.forward()`, `GPT.forward_logits()`, training, cache/adaptation logic, and eval scoring untouched in Stages 1-3.

## Success Metrics / Kill Criteria
- **Success metrics from the memo:**
  - artifact stays within `16,000,000` bytes,
  - `val_bpb` is no worse than about `0.002` relative to the current Int4 MLP path or is better,
  - no meaningful eval-time penalty beyond one-time dequantization work.
- **Kill criteria from the memo:**
  - rotation overhead pushes the artifact over budget,
  - `val_bpb` regresses by more than about `0.005`,
  - 512-dim behavior is too far from the assumptions needed for the quantization gains to matter.

## Repo Strategy
- The memo's recommendation for actual Stage 1 work was **same repo + branch**, not a separate repo.
- The reasoning was straightforward: the change would be isolated to `train_gpt.py`, easy to diff against `main`, and easy to delete if the evidence turns negative.

## Current Evidence
- **Synthetic orthogonal-init probe only:**
  - `TQ-Centroid` won `19/20` tensors individually by MSE.
  - Aggregate MSE was still worse because one tensor produced a large outlier.
  - `TQ-Hybrid` did not beat the baseline aggregate.
- No target-environment `val_bpb` evidence exists for TurboQuant on this branch.

## Not Yet Proven
- Whether lower quantization MSE would transfer to better or even neutral `val_bpb`.
- Whether the actual serialized artifact would stay under budget after wiring Stage 1 into `train_gpt.py`.
- Whether the dequantization-side changes would remain as cheap in practice as the memo expects.

## Risks / Open Questions
- **MSE-to-BPB transfer risk:** better weight reconstruction may still fail to improve scored BPB.
- **Budget risk:** even a shared rotation matrix is expensive enough that the quantization gain must pay for it.
- **Scope creep risk:** if residual correction or QAT become necessary, the experiment stops being a small serialization-path probe.

## Next Concrete Step
Use **Runpod** (or equivalent GPU staging) for target-environment evidence; judge Stage 1 serialization-path A/B by artifact size, runtime, and `val_bpb` instead of probe MSE alone.
