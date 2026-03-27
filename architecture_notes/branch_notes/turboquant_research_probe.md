# Branch: research/turboquant-probe

## Purpose
- Isolated research branch for empirical validation of the TurboQuant algorithm (arXiv:2504.19874).
- Testing the feasibility of unitary rotation + Lloyd-Max centroids on trained model weights.
- Derived from: [turboquant_research_memo.md](file:///Users/jmoncayopursuit.org/.gemini/antigravity/brain/1cf0e198-d0e0-43e8-aa33-a4e3ccbed7e0/turboquant_research_memo.md)

## Verified Now
- **Commit Tip:** `490daa0`
- **Implemented Now:** `turboquant_mse_probe.py` script with the following modules:
  - **Stage 1 (Serialization-Path Only):** Hadamard/Unitary rotation + baseline block-scaling.
  - **Stage 2 (Full TQ):** Pure-numpy Lloyd-Max iterative centroid solver.

## Architecture / Method (Faithful Relay of Memo)
- **Feasibility Rationale:** Per arXiv:2504.19874, unitary rotation smooths the weight distribution, potentially lowering quantization noise floor below the Int4 baseline.
- **Engineering Trade-off:** Memo derives a **~500KB - 800KB** artifact overhead for the rotation matrix, requiring a **~0.015 - 0.02 BPB** reduction on trained weights to justify submission.
- **Hybrid Strategy:** Testing the transfer of KV-cache techniques to static model weights.

## What This Is Not
- **Not yet an Int4-verified BPB gain:** The 0.015 BPB gain is a **hypothesis** based on the 11% MSE distortion reduction target.
- **Not for real-time inference:** The current Lloyd-Max implementation is optimized for the serialization-path budget, not inference-path latency.

## Measurements
- **Synthetic Orthogonal-Init Probe:**
  - **Individual Tensor Success:** TQ-Centroid won 19/20 tensor trials by MSE.
  - **Aggregate Failure:** One outlier tensor caused the aggregate MSE to exceed the baseline.
  - **TQ-Hybrid Result:** Baseline block-scaling + rotation failed to beat naive Int4 on synthetic weights.
- **Trained-Weight Evidence:** Pending (awaiting local `final_model.pt` access).

## Risks / Open Questions
- **MSE-to-BPB Transfer:** The most critical risk is whether the 11% MSE gain from TQ results in a measurable compression win on the final 13.8MB artifact.
- **Outlier Sensitivity:** The Stage 2 Lloyd-Max solver requires further refinement to handle the single-outlier failure seen in the synthetic aggregate.

## Next Concrete Step
Run the `turboquant_mse_probe.py` against the `final_model.pt` weights generated in Colab.
