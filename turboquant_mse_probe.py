"""
TurboQuant MSE Measurement Probe
================================
Throwaway script to answer: does rotation + Lloyd-Max centroid quantization
produce lower MSE than the baseline block-scaled Int4 quantization at the
same bit-width?

This script does NOT modify the parameter-golf repo.
It does NOT create branches or commits.

Usage:
    python /tmp/turboquant_mse_probe.py [--checkpoint path/to/final_model.pt]

If no checkpoint is provided, it initializes the model with the same
orthogonal init used in train_gpt.py to get realistic weight distributions.
"""

import argparse
import io
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

# ============================================================================
# BASELINE QUANTIZATION (extracted from train_gpt.py)
# ============================================================================

def quantize_intN_blocked_baseline(t: Tensor, clip_range: int = 7, block_size: int = 128):
    """Baseline Int4 blocked quantization from train_gpt.py L286-308."""
    t32 = t.float()
    if t32.ndim == 2:
        orig_shape = t32.shape
        pad_len = (block_size - (orig_shape[1] % block_size)) % block_size
        if pad_len > 0:
            t32 = F.pad(t32, (0, pad_len))
        t32_blocked = t32.view(-1, block_size)
        block_max = t32_blocked.abs().amax(dim=1)
        scale = (block_max / clip_range).clamp_min(1e-12).to(torch.float16)
        scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
        q_blocked = torch.clamp(
            torch.round(t32_blocked / scale.float()[:, None]),
            -(clip_range + 1), clip_range
        ).to(torch.int8)
        q = q_blocked.view(orig_shape[0], -1)
        if pad_len > 0:
            q = q[:, :orig_shape[1]]
        return q, scale, orig_shape
    raise ValueError("Only 2D tensors supported")


def dequantize_intN_blocked_baseline(q: Tensor, scale: Tensor, orig_shape: tuple, block_size: int = 128):
    """Baseline dequantization."""
    pad_len = (block_size - (orig_shape[1] % block_size)) % block_size
    if pad_len > 0:
        q = F.pad(q, (0, pad_len))
    q_blocked = q.contiguous().view(-1, block_size)
    return (q_blocked.float() * scale.float()[:, None]).view(orig_shape[0], -1)[:, :orig_shape[1]]


# ============================================================================
# TURBOQUANT ROTATION + LLOYD-MAX CENTROID QUANTIZATION
# ============================================================================

def generate_random_rotation(d: int, seed: int = 42) -> Tensor:
    """
    Generate a d×d random orthogonal rotation matrix via QR decomposition
    of a random Gaussian matrix. This is the TurboQuant approach (§3.1).
    """
    rng = torch.Generator().manual_seed(seed)
    G = torch.randn(d, d, generator=rng)
    Q, R = torch.linalg.qr(G)
    # Ensure proper rotation (det = +1) by fixing sign ambiguity
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.float()


def compute_lloyd_max_centroids(num_levels: int, d: int):
    """
    Compute Lloyd-Max optimal centroids for the Beta distribution of
    coordinates of a random rotation of a unit-norm vector.
    
    In high dimensions, each coordinate of Π·x (where x is unit-norm)
    follows approximately N(0, 1/d). We solve the 1D k-means for this
    distribution numerically.
    
    For b=2 bits (4 levels), the paper states (§3.1):
        centroids ≈ {±0.453/√d, ±1.51/√d}
    
    For b=1 bit (2 levels):
        centroids ≈ {±√(2/π)/√d}
    
    We implement a general Lloyd-Max iteration for N(0, 1/d).
    """
    sigma = 1.0 / math.sqrt(d)
    
    # Initialize centroids uniformly
    # For symmetric distributions, we only need to compute positive half
    half_levels = num_levels // 2
    
    if num_levels == 2:
        # Analytical: ±√(2/π) · σ
        c = math.sqrt(2.0 / math.pi) * sigma
        return torch.tensor([-c, c], dtype=torch.float32)
    
    if num_levels == 4:
        # Paper §3.1: ±0.453/√d, ±1.51/√d
        c1 = 0.4528 * sigma  # inner centroids
        c2 = 1.5104 * sigma  # outer centroids
        return torch.tensor([-c2, -c1, c1, c2], dtype=torch.float32)
    
    if num_levels == 8:
        # Lloyd-Max for N(0, σ²) with 8 levels via iteration
        return _lloyd_max_gaussian(8, sigma, max_iter=200)
    
    if num_levels == 16:
        # 4-bit: Lloyd-Max for N(0, σ²) with 16 levels
        return _lloyd_max_gaussian(16, sigma, max_iter=200)
    
    return _lloyd_max_gaussian(num_levels, sigma, max_iter=200)


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x):
    """Standard normal CDF via math.erf."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_ppf(p, sigma=1.0):
    """Approximate inverse CDF (quantile function) for N(0, σ²) using bisection."""
    result = np.zeros_like(p, dtype=np.float64)
    for i, pi in enumerate(p):
        lo, hi = -8.0 * sigma, 8.0 * sigma
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if _norm_cdf(mid / sigma) < pi:
                lo = mid
            else:
                hi = mid
        result[i] = (lo + hi) / 2.0
    return result


def _lloyd_max_gaussian(num_levels: int, sigma: float, max_iter: int = 200) -> Tensor:
    """
    General Lloyd-Max iteration for N(0, σ²).
    Uses the closed-form conditional expectation for Gaussian:
        E[X | a < X < b] = σ² · (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
    where φ is the standard normal PDF and Φ is the CDF.
    Pure numpy implementation — no scipy required.
    """
    # Initialize with uniform quantile spacing
    quantiles = np.linspace(0.5 / num_levels, 1.0 - 0.5 / num_levels, num_levels)
    centroids = _norm_ppf(quantiles, sigma)
    
    for _ in range(max_iter):
        # Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.concatenate([
            [-np.inf],
            (centroids[:-1] + centroids[1:]) / 2.0,
            [np.inf]
        ])
        
        # Recompute centroids as conditional expectations
        new_centroids = np.zeros_like(centroids)
        for i in range(num_levels):
            a, b = boundaries[i], boundaries[i + 1]
            a_std = a / sigma if np.isfinite(a) else -1e10
            b_std = b / sigma if np.isfinite(b) else 1e10
            
            prob = _norm_cdf(b_std) - _norm_cdf(a_std)
            if prob < 1e-15:
                new_centroids[i] = (a + b) / 2.0 if np.isfinite(a) and np.isfinite(b) else centroids[i]
            else:
                # E[X | a < X < b] for X ~ N(0, σ²)
                new_centroids[i] = sigma * (_norm_pdf(a_std) - _norm_pdf(b_std)) / prob
        
        if np.max(np.abs(new_centroids - centroids)) < 1e-12:
            break
        centroids = new_centroids
    
    return torch.tensor(centroids, dtype=torch.float32)


def quantize_turboquant_mse(t: Tensor, rotation: Tensor, centroids: Tensor):
    """
    TurboQuant MSE quantization (§3.1 Algorithm 1):
    1. Normalize each row to unit norm, store norms
    2. Rotate: y = Π · x
    3. Quantize each coordinate to nearest centroid
    4. Store centroid indices
    """
    t32 = t.float()
    assert t32.ndim == 2, "Only 2D tensors supported"
    
    # Step 1: Store row norms, normalize to unit norm
    row_norms = t32.norm(dim=1, keepdim=True).clamp_min(1e-12)
    t_normalized = t32 / row_norms
    
    # Step 2: Rotate
    # t_normalized is (rows, d), rotation is (d, d)
    # rotated = t_normalized @ rotation.T  (each row gets rotated)
    rotated = t_normalized @ rotation.T
    
    # Step 3: Quantize to nearest centroid
    # centroids is (num_levels,)
    # For each element, find the closest centroid
    diffs = rotated.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0)  # (rows, d, num_levels)
    indices = diffs.abs().argmin(dim=-1)  # (rows, d)
    
    return indices.to(torch.uint8), row_norms.squeeze(1).to(torch.float16), centroids


def dequantize_turboquant_mse(indices: Tensor, row_norms: Tensor, centroids: Tensor, rotation: Tensor):
    """
    TurboQuant MSE dequantization:
    1. Look up centroids from indices
    2. Inverse rotate: x̃ = Πᵀ · ỹ
    3. Rescale by stored norms
    """
    # Step 1: Centroid lookup
    reconstructed_rotated = centroids[indices.long()].float()  # (rows, d)
    
    # Step 2: Inverse rotation
    reconstructed = reconstructed_rotated @ rotation  # rotation is orthogonal, so Πᵀ = Π⁻¹
    
    # Step 3: Rescale
    reconstructed = reconstructed * row_norms.float().unsqueeze(1)
    
    return reconstructed


# ============================================================================
# MEASUREMENT FUNCTIONS
# ============================================================================

def measure_mse(original: Tensor, reconstructed: Tensor) -> dict:
    """Compute per-tensor and aggregate MSE metrics."""
    diff = (original.float() - reconstructed.float())
    mse = (diff ** 2).mean().item()
    max_abs_err = diff.abs().max().item()
    rel_mse = mse / (original.float() ** 2).mean().item() if (original.float() ** 2).mean().item() > 0 else float('inf')
    return {
        "mse": mse,
        "max_abs_err": max_abs_err,
        "relative_mse": rel_mse,
        "rmse": math.sqrt(mse),
    }


def measure_artifact_overhead(rotation: Tensor, compress: bool = True) -> dict:
    """Measure the byte cost of storing the rotation matrix."""
    rot_fp16 = rotation.to(torch.float16)
    
    # Uncompressed size
    uncompressed = rot_fp16.numel() * rot_fp16.element_size()
    
    # Compressed size (lzma, matching baseline compressor)
    import lzma
    buf = io.BytesIO()
    torch.save(rot_fp16, buf)
    raw = buf.getvalue()
    compressed = len(lzma.compress(raw, preset=6))
    
    # Also try zstd if available
    zstd_compressed = None
    try:
        import zstandard
        zstd_compressed = len(zstandard.ZstdCompressor(level=22).compress(raw))
    except ImportError:
        pass
    
    return {
        "uncompressed_bytes": uncompressed,
        "lzma_compressed_bytes": compressed,
        "zstd_compressed_bytes": zstd_compressed,
        "raw_torch_bytes": len(raw),
    }


def bits_per_weight(method: str, tensor_shape: tuple, block_size: int = 128) -> float:
    """Estimate effective bits per weight for each method."""
    rows, cols = tensor_shape
    
    if method == "baseline_int4":
        # Int4: 4 bits per weight + scale overhead
        # Scales: one float16 per block of 128
        num_blocks = math.ceil(cols / block_size) * rows
        weight_bits = rows * cols * 4  # 4 bits per element (stored as packed nibbles)
        scale_bits = num_blocks * 16  # float16 scales
        total_bits = weight_bits + scale_bits
        return total_bits / (rows * cols)
    
    elif method == "turboquant_2bit":
        # 2-bit centroids (4 levels): 2 bits per coordinate
        # Plus: row norms (float16 per row) + rotation matrix (amortized)
        weight_bits = rows * cols * 2
        norm_bits = rows * 16
        return (weight_bits + norm_bits) / (rows * cols)
    
    elif method == "turboquant_4bit":
        # 4-bit centroids (16 levels): 4 bits per coordinate
        weight_bits = rows * cols * 4
        norm_bits = rows * 16
        return (weight_bits + norm_bits) / (rows * cols)
    
    return float('nan')


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def create_model_weights(checkpoint_path: str = None):
    """
    Load or generate weight tensors matching the Parameter Golf model's
    MLP layers. If no checkpoint, use orthogonal init to match train_gpt.py.
    """
    model_dim = 512
    mlp_mult = 4.0
    num_layers = 10
    hidden = int(mlp_mult * model_dim)  # 2048
    
    tensors = {}
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        sd = torch.load(checkpoint_path, map_location="cpu")
        for name, param in sd.items():
            if ".mlp." in name and param.ndim == 2:
                tensors[name] = param
        if tensors:
            print(f"  Loaded {len(tensors)} MLP weight tensors from checkpoint")
            return tensors
        print("  No MLP tensors found in checkpoint, falling back to synthetic")
    
    print(f"Generating synthetic MLP weights (orthogonal init, {num_layers} layers)")
    print(f"  model_dim={model_dim}, hidden={hidden}")
    
    torch.manual_seed(42)
    for i in range(num_layers):
        # fc: (hidden, model_dim) = (2048, 512)
        fc = torch.empty(hidden, model_dim)
        torch.nn.init.orthogonal_(fc, gain=1.0)
        tensors[f"blocks.{i}.mlp.fc.weight"] = fc
        
        # proj: (model_dim, hidden) = (512, 2048)  — zero init in train_gpt.py
        # But zero-init tensors have nothing to quantize, so use orthogonal
        # with the scaling factor from train_gpt.py
        proj = torch.empty(model_dim, hidden)
        torch.nn.init.orthogonal_(proj, gain=1.0)
        proj *= 1.0 / math.sqrt(2 * num_layers)
        tensors[f"blocks.{i}.mlp.proj.weight"] = proj
    
    return tensors


def run_experiment(tensors: dict, bits: int = 4):
    """Run the full comparison experiment."""
    d = None
    
    # Determine model dim from tensors
    for name, t in tensors.items():
        if "fc" in name:
            d = t.shape[1]  # input dim
            break
        elif "proj" in name:
            d = t.shape[0]  # output dim
            break
    
    if d is None:
        d = 512
    
    print(f"\n{'='*80}")
    print(f"TURBOQUANT MSE MEASUREMENT PROBE")
    print(f"{'='*80}")
    print(f"Model dimension: {d}")
    print(f"Target bit-width: {bits}")
    print(f"Number of MLP tensors: {len(tensors)}")
    print(f"Total MLP parameters: {sum(t.numel() for t in tensors.values()):,}")
    
    # Generate rotation matrix
    print(f"\n--- Rotation Matrix ---")
    t_rot_start = time.perf_counter()
    rotation = generate_random_rotation(d, seed=42)
    t_rot_gen = time.perf_counter() - t_rot_start
    print(f"Generated {d}×{d} rotation matrix in {t_rot_gen*1000:.1f}ms")
    
    # Verify orthogonality
    orth_err = (rotation @ rotation.T - torch.eye(d)).abs().max().item()
    print(f"Orthogonality error: {orth_err:.2e}")
    
    # Measure rotation storage cost
    rot_overhead = measure_artifact_overhead(rotation)
    print(f"Rotation matrix storage:")
    print(f"  Uncompressed (float16): {rot_overhead['uncompressed_bytes']:,} bytes ({rot_overhead['uncompressed_bytes']/1024:.1f} KB)")
    print(f"  lzma compressed:        {rot_overhead['lzma_compressed_bytes']:,} bytes ({rot_overhead['lzma_compressed_bytes']/1024:.1f} KB)")
    if rot_overhead['zstd_compressed_bytes']:
        print(f"  zstd compressed:         {rot_overhead['zstd_compressed_bytes']:,} bytes ({rot_overhead['zstd_compressed_bytes']/1024:.1f} KB)")
    
    # Compute centroids
    num_levels = 2 ** bits
    clip_range = num_levels // 2 - 1  # 7 for 4-bit, 1 for 2-bit
    centroids = compute_lloyd_max_centroids(num_levels, d)
    print(f"\n--- Lloyd-Max Centroids ({num_levels} levels, for N(0, 1/{d})) ---")
    print(f"  Centroids: {[f'{c:.6f}' for c in centroids.tolist()]}")
    
    # Run per-tensor comparison: 3 methods
    print(f"\n--- Per-Tensor MSE Comparison (3 methods) ---")
    print(f"{'Tensor':<40} {'Shape':<15} {'Baseline':>12} {'TQ-Centroid':>12} {'TQ-Hybrid':>12} {'Best':>12}")
    print("-" * 105)
    
    baseline_total_se = 0.0
    tq_centroid_total_se = 0.0
    tq_hybrid_total_se = 0.0
    total_elements = 0
    method_wins = {"Baseline": 0, "TQ-Centroid": 0, "TQ-Hybrid": 0}
    
    for name, t in sorted(tensors.items()):
        n_elem = t.numel()
        
        # --- Method 1: Baseline Int4 blocked (no rotation) ---
        q_b, s_b, orig_shape = quantize_intN_blocked_baseline(t, clip_range=clip_range, block_size=128)
        recon_b = dequantize_intN_blocked_baseline(q_b, s_b, orig_shape, block_size=128)
        mse_b = ((t.float() - recon_b.float()) ** 2).mean().item()
        
        # --- Method 2: TurboQuant rotation + Lloyd-Max centroids ---
        if t.shape[1] == d:
            idx, norms, _ = quantize_turboquant_mse(t, rotation, centroids)
            recon_tq_c = dequantize_turboquant_mse(idx, norms, centroids, rotation)
        elif t.shape[0] == d:
            idx, norms, _ = quantize_turboquant_mse(t.T, rotation, centroids)
            recon_tq_c = dequantize_turboquant_mse(idx, norms, centroids, rotation).T
        else:
            recon_tq_c = recon_b
        recon_tq_c = recon_tq_c[:t.shape[0], :t.shape[1]]
        mse_c = ((t.float() - recon_tq_c.float()) ** 2).mean().item()
        
        # --- Method 3: TQ-Hybrid = Rotation + baseline block-scaled quantizer ---
        # Apply rotation to rows, then quantize with baseline block-scaling
        # This tests whether rotation alone helps the existing quantizer
        if t.shape[1] == d:
            row_norms = t.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
            t_normalized = t.float() / row_norms
            t_rotated = t_normalized @ rotation.T
            # Scale back by row norms so the block quantizer sees realistic magnitudes
            t_rotated_scaled = t_rotated * row_norms
            q_h, s_h, orig_shape_h = quantize_intN_blocked_baseline(t_rotated_scaled, clip_range=clip_range, block_size=128)
            recon_h_rot = dequantize_intN_blocked_baseline(q_h, s_h, orig_shape_h, block_size=128)
            # Inverse rotation
            recon_h_rot_norm = recon_h_rot / row_norms
            recon_tq_h = (recon_h_rot_norm @ rotation).float() * row_norms
            recon_tq_h = recon_tq_h[:t.shape[0], :t.shape[1]]
        elif t.shape[0] == d:
            tt = t.T
            row_norms = tt.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
            t_normalized = tt.float() / row_norms
            t_rotated = t_normalized @ rotation.T
            t_rotated_scaled = t_rotated * row_norms
            q_h, s_h, orig_shape_h = quantize_intN_blocked_baseline(t_rotated_scaled, clip_range=clip_range, block_size=128)
            recon_h_rot = dequantize_intN_blocked_baseline(q_h, s_h, orig_shape_h, block_size=128)
            recon_h_rot_norm = recon_h_rot / row_norms
            recon_tq_h = ((recon_h_rot_norm @ rotation) * row_norms).T.float()
            recon_tq_h = recon_tq_h[:t.shape[0], :t.shape[1]]
        else:
            recon_tq_h = recon_b
        mse_h = ((t.float() - recon_tq_h.float()) ** 2).mean().item()
        
        # Determine winner
        mses = {"Baseline": mse_b, "TQ-Centroid": mse_c, "TQ-Hybrid": mse_h}
        best = min(mses, key=mses.get)
        method_wins[best] += 1
        
        print(f"{name:<40} {str(list(t.shape)):<15} {mse_b:>12.10f} {mse_c:>12.10f} {mse_h:>12.10f} {best:>12}")
        
        baseline_total_se += mse_b * n_elem
        tq_centroid_total_se += mse_c * n_elem
        tq_hybrid_total_se += mse_h * n_elem
        total_elements += n_elem
    
    # Aggregate results
    baseline_avg_mse = baseline_total_se / total_elements
    tq_centroid_avg_mse = tq_centroid_total_se / total_elements
    tq_hybrid_avg_mse = tq_hybrid_total_se / total_elements
    
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*80}")
    print(f"Total MLP elements:          {total_elements:,}")
    print(f"Baseline avg MSE:            {baseline_avg_mse:.10f}")
    print(f"TQ-Centroid avg MSE:         {tq_centroid_avg_mse:.10f}  ({tq_centroid_avg_mse/baseline_avg_mse:.4f}x baseline)")
    print(f"TQ-Hybrid avg MSE:           {tq_hybrid_avg_mse:.10f}  ({tq_hybrid_avg_mse/baseline_avg_mse:.4f}x baseline)")
    print(f"\nTensor wins — Baseline: {method_wins['Baseline']}, TQ-Centroid: {method_wins['TQ-Centroid']}, TQ-Hybrid: {method_wins['TQ-Hybrid']}")
    
    print(f"\n--- Artifact Size Impact ---")
    print(f"Rotation matrix overhead (lzma): {rot_overhead['lzma_compressed_bytes']:,} bytes")
    print(f"16MB budget:                     16,000,000 bytes")
    print(f"Overhead as % of budget:         {rot_overhead['lzma_compressed_bytes']/16_000_000*100:.2f}%")
    
    # Dimension suitability check
    print(f"\n--- Dimension Suitability Check (d={d}) ---")
    test_vec = torch.randn(1000, d)
    test_vec = test_vec / test_vec.norm(dim=1, keepdim=True)
    rotated_test = test_vec @ rotation.T
    coord_mean = rotated_test.mean().item()
    coord_var = rotated_test.var().item()
    expected_var = 1.0 / d
    print(f"Expected coordinate variance (1/d): {expected_var:.6f}")
    print(f"Observed coordinate variance:       {coord_var:.6f}")
    print(f"Variance ratio (observed/expected):  {coord_var/expected_var:.4f}")
    corr_01 = torch.corrcoef(rotated_test[:, :2].T)[0, 1].item()
    print(f"Correlation between coords 0 and 1: {corr_01:.6f}")
    
    # GO / NO-GO SIGNAL
    print(f"\n{'='*80}")
    print(f"GO / NO-GO ASSESSMENT")
    print(f"{'='*80}")
    
    go_signals = []
    nogo_signals = []
    
    best_tq_ratio = min(tq_centroid_avg_mse, tq_hybrid_avg_mse) / max(baseline_avg_mse, 1e-30)
    best_method = "TQ-Centroid" if tq_centroid_avg_mse < tq_hybrid_avg_mse else "TQ-Hybrid"
    
    if best_tq_ratio < 0.95:
        go_signals.append(f"MSE improvement: {best_method} is {(1-best_tq_ratio)*100:.1f}% better than baseline")
    elif best_tq_ratio > 1.05:
        nogo_signals.append(f"MSE regression: best TQ method ({best_method}) is {(best_tq_ratio-1)*100:.1f}% worse than baseline")
    else:
        go_signals.append(f"MSE roughly comparable ({best_method} ratio={best_tq_ratio:.4f}x)")
    
    if rot_overhead['lzma_compressed_bytes'] > 800_000:
        nogo_signals.append(f"Rotation matrix overhead too large: {rot_overhead['lzma_compressed_bytes']:,} bytes > 800KB")
    else:
        go_signals.append(f"Rotation matrix overhead acceptable: {rot_overhead['lzma_compressed_bytes']:,} bytes")
    
    if abs(coord_var / expected_var - 1.0) > 0.1:
        nogo_signals.append(f"Coordinate variance deviates >10% from expected (d={d} may be too small)")
    else:
        go_signals.append(f"Coordinate distribution matches N(0,1/d) well at d={d}")
    
    for s in go_signals:
        print(f"  ✅ {s}")
    for s in nogo_signals:
        print(f"  ❌ {s}")
    
    if nogo_signals:
        print(f"\nVERDICT: CONDITIONAL — {len(nogo_signals)} concern(s) need resolution before Stage 1")
    else:
        print(f"\nVERDICT: GO — preliminary evidence supports Stage 1 implementation")
    
    print(f"\nCAVEAT: These measurements use {'synthetic' if not any('checkpoint' in str(v) for v in tensors.values()) else 'trained'} weights.")
    print(f"         Final go/no-go requires trained checkpoint + H100 val_bpb measurement.")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant MSE Measurement Probe")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to final_model.pt trained checkpoint")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 4],
                        help="Bit-width to test (default: 4)")
    args = parser.parse_args()
    
    tensors = create_model_weights(args.checkpoint)
    run_experiment(tensors, bits=args.bits)


if __name__ == "__main__":
    main()
