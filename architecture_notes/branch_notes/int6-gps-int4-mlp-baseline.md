# Int6 GPS + Int4 MLP candidate

Git branch: **`int6-gps-int4-mlp`** (Int6 attention / GPS-style slots + Int4 MLP stack).

## Purpose
- Canonical candidate branch for the current Parameter Golf submission line.
- Holds the mixed-precision training/export pipeline we would actually want to reproduce on target hardware.
- Exists to answer "does the current candidate really run and score as expected on challenge hardware?"

## Verified Now
- `train_gpt.py` already contains the core candidate stack:
  - Muon with Newton-Schulz orthogonalized matrix updates for matrix-shaped parameters.
  - Orthogonal initialization for large non-zero `nn.Linear` weights, with extra projection scaling.
  - QAT clip ranges that map to the current mixed-precision design: Int4 MLP weights and Int6 attention/bigram weights.
  - `mixed_quantize_int6()` / `dequantize_mixed_int6()` export roundtrip and `final_model.mixed.ptz` artifact path.
  - SWA application plus 3% magnitude pruning before serialization.
  - `BayesianBackoffCache` inside `eval_val_sliding_cached()` for evaluation-side mixing (see [BayesianBackoffCache_TTAdapter.md](./BayesianBackoffCache_TTAdapter.md) for the extended **cache + TestTimeAdapter** eval-time line; rename upstream repo to match).
- No local Runpod/H100 artifact or benchmark log is committed for this branch.

## Feasibility Summary
- **Muon is already integrated, not hypothetical.** The branch already routes matrix-shaped transformer parameters through `Muon` and keeps scalar/embedding-style parameters on `AdamW`. This is the same split that prior local records used successfully.
- **Muon + orthogonal init is a coherent pairing.** Local historical notes in [2026-03-19_smeargate_orthoinit_muonwd/README.md](/Users/jmoncayopursuit.org/Desktop/parameter-golf/records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md) explain the engineering rationale: Muon orthogonalizes updates, while orthogonal init gives large linear weights well-conditioned starting geometry, so early steps are spent learning rather than correcting poor initial singular values.
- **The quantization path is already end-to-end.** This branch is not waiting on a future quantization implementation; QAT clip ranges are attached to the actual layers, fake quantization is used during training, and the export path already serializes a compressed mixed-precision artifact.
- **The remaining uncertainty is execution evidence, not missing plumbing.** The main unknown is whether the current coded stack reproduces under target wall-clock and memory constraints, not whether the branch has the necessary components.
- **Eval-side cache feasibility is rule-based, not officially blessed.** The current reading of the README is that using already-graded tokens for backward-looking mixing is acceptable, but there is no explicit organizer ruling attached to this branch note.

## Current Evidence
- **External measurement only:** a friend-run result reported `Post-EMA = 1.1193` at `13.8MB`.
- That result is useful signal, but it is not yet backed by a committed artifact, log, or `records/` entry in this branch.
- There are no local Runpod/H100 measurements committed here yet.

## Not Yet Proven
- Whether the external `1.1193 / 13.8MB` result is reproducible from this exact branch.
- Whether the current cached sliding evaluation path fits comfortably inside the 10-minute evaluation budget on target hardware.
- Whether current README-compatible cache usage matches organizer intent in a stricter ruling.

## Risks / Open Questions
- **Reproducibility risk:** external numbers may depend on uncommitted launch details or environment differences.
- **Runtime risk:** cached sliding evaluation may help BPB but still fail the wall-clock budget if not measured carefully.
- **Artifact risk:** final compressed size is promising externally, but not yet confirmed in a local committed run.

## Next Concrete Step
Run and save one reproducible target-hardware baseline result from this branch, including artifact size, wall-clock, and `val_bpb`, so the branch stops depending on external testimony.
