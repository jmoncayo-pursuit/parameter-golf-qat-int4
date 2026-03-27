# Branch: research/turboquant-probe

## Purpose
- Isolated research branch for validating Lloyd-Max quantization on trained weights.
- Testing the transfer of TurboQuant-style rotation strategies to model weights.

## Verified Now
- **Commit Tip:** `490daa0`
- **Implemented Now:** `turboquant_mse_probe.py` with pure-numpy Lloyd-Max solvers and rotation logic.

## Architecture / Method
- **TQ-Centroid:** Iterative centroid optimization (Lloyd-Max) for optimal discrete level mapping.
- **TQ-Hybrid:** Combined unitary rotation (Hadamard/Random) with block-scaling.
- **Artifact Overhead:** Rotation matrix estimated at `~471KB lzma`.

## What This Is Not
- **Not a Candidate Model:** Research use only; not intended for submission script.
- **Not a Trained-Weight Result:** Measurements are currently based on synthetic weights only.

## Measurements
- **Synthetic Orthogonal-Init Probe:**
  - `d=512` variance behavior is confirmed within acceptable bounds.
  - `TQ-Centroid`: Improved MSE for 19 out of 20 individual tensors, but **aggregate MSE was worse** due to one extreme outlier.
  - `TQ-Hybrid`: Aggregate MSE was worse than the naive Int4 baseline.
- **No Trained-Weight Evidence yet.**

## Risks / Open Questions
- **Aggregation Failure:** Risk that Lloyd-Max optimizations on single tensors do not reliably lower aggregate model BPB due to outlier sensitivity.
- **MSE-to-BPB Transfer:** No evidence exists yet that lower synthetic MSE translates to lower BPB on trained weights.
- **Artifact Overhead:** Compressed size of the rotation matrix (471KB) may exceed the gain from lower distortion.

## Next Concrete Step
Run the `turboquant_mse_probe.py` against the `final_model.pt` weights from the Colab T4 run to check for trained-weight transfer.
