# Main Candidate Branch

## Purpose
- Canonical candidate branch for the Parameter Golf submission.
- Integration test of Int4/Int6 mixed-precision and the Muon optimizer.
- This branch exists to measure the current candidate on target hardware.

## Verified Now
- **Commit Tip:** `afd88d9`
- **Coded Now:** `train_gpt.py` contains the implementation for Int4/Int6 packing, Muon optimizer, and SWA warmdown.
- **Execution Status:** ZERO local Runpod/H100 runs have been completed.

## Planned Only
- **Runpod/H100 Validation:** Initial single-GPU stability test on target hardware.
- **H100 Cluster Execution:** Final 8xH100 training/evaluation run.
- **Local BPB Benchmark:** Establishing a reproducible BPB floor on challenge hardware (1-10 min evaluation window).

## Architecture / Method
- **Quantization:** Blocked Int4 (MLP) and Int6 (Attention) mixed-precision scheme.
- **Optimizer:** Muon for matrix parameters; AdamW for scalar/embedding weights.
- **Post-Weights:** 3% magnitude pruning applied pre-serialization; SWA enabled for final warmdown phase.

## What This Is Not
- **Not a submission baseline:** Results are hypotheses until the "Planned Only" hardware validation phase is complete.
- **Not locally verified at scale.**

## Measurements
- **No hardware benchmarks available.**
- **External Measurement (Reported):** `Post-EMA = 1.1193, 13.8MB` (Reported; not a local artifact; not verified by any local Runpod log).

## Risks / Open Questions
- **Hardware Failure:** It is not yet verified if the script will run without OOM or compatibility errors on challenge-grade H100s.
- **Reproduction Risk:** Unknown if the reported `1.1193` BPB is reproducible under challenge wall-clock constraints.

## Next Concrete Step
Execute the first local Runpod test to establish an actual BPB floor.
