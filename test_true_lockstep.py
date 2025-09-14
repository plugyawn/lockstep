#!/usr/bin/env python
"""
Test true lockstep R/A decoding where both blocks fill in tandem.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from src.core.lockstep_controller import LockstepController, LockstepCfg
from src.core.gif_viz import create_decoding_gif, visualize_block_statistics, create_fill_order_heatmap


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self):
        self.mask_token_id = 103
        self.mask_token = "[MASK]"
        self.vocab_size = 1000

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        text = ""
        for tid in token_ids:
            if tid == self.mask_token_id and not skip_special_tokens:
                text += "[M]"
            else:
                text += f"T{tid%100:02d} "
        return text.strip()


def test_true_lockstep(mode="gauss_seidel"):
    """Test true lockstep behavior where R and A fill in tandem."""

    print("=" * 70)
    print(f"TRUE LOCKSTEP TEST - Mode: {mode}")
    print("=" * 70)

    # Setup
    batch_size = 2
    seq_len = 80
    vocab_size = 1000
    mask_token_id = 103

    tokenizer = MockTokenizer()

    # Configure for true lockstep
    cfg = LockstepCfg(
        mode=mode,
        tau_r=0.65,  # Lower threshold for R to start filling
        tau_a=0.70,  # Slightly higher for A (but close to R)
        max_fill_frac=0.20,  # Fill 20% per step to see gradual progress
        halo=2,
        r_span=(15, 40),  # R block: 25 tokens
        a_span=(45, 70),  # A block: 25 tokens
    )

    controller = LockstepController(cfg, tokenizer)

    # Initialize sequence
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    x[:, cfg.r_span[0]:cfg.r_span[1]] = mask_token_id  # Mask R
    x[:, cfg.a_span[0]:cfg.a_span[1]] = mask_token_id  # Mask A

    print(f"Configuration:")
    print(f"  R block: positions {cfg.r_span} ({cfg.r_span[1]-cfg.r_span[0]} tokens)")
    print(f"  A block: positions {cfg.a_span} ({cfg.a_span[1]-cfg.a_span[0]} tokens)")
    print(f"  Thresholds: τ_R={cfg.tau_r}, τ_A={cfg.tau_a}")
    print(f"  Max fill per step: {cfg.max_fill_frac*100:.0f}%")

    # Track progress
    num_steps = 25
    output_history = []
    r_fill_history = []
    a_fill_history = []

    print(f"\nSimulating {num_steps} steps...")
    print(f"{'Step':>4} {'R%':>6} {'A%':>6} {'R_new':>6} {'A_new':>6} {'Status':>20}")
    print("-" * 60)

    for step in range(num_steps):
        # Generate logits with gradually increasing confidence
        base_confidence = 2.0 + (step * 0.3)
        logits = torch.randn(batch_size, seq_len, vocab_size) * (2.0 - step * 0.05)

        # Add confidence to both R and A blocks simultaneously
        # R block gets slightly earlier/stronger signal
        if step >= 2:
            # R block confidence
            r_start, r_end = cfg.r_span
            num_r_confident = min(3 + step//2, 10)
            for _ in range(num_r_confident):
                pos = torch.randint(r_start, r_end, (1,)).item()
                if x[0, pos] == mask_token_id:  # Check if still masked
                    for b in range(batch_size):
                        target = torch.randint(0, vocab_size, (1,)).item()
                        logits[b, pos, target] += base_confidence

        # A block confidence (slightly delayed but similar strength)
        if step >= 3:
            a_start, a_end = cfg.a_span
            num_a_confident = min(2 + step//2, 10)
            for _ in range(num_a_confident):
                pos = torch.randint(a_start, a_end, (1,)).item()
                if x[0, pos] == mask_token_id:
                    for b in range(batch_size):
                        target = torch.randint(0, vocab_size, (1,)).item()
                        # A gets confidence based on step progress
                        logits[b, pos, target] += base_confidence * 0.9

        # Process through controller
        controller.logits_hook(logits, {})
        x_old = x.clone()
        x = controller.tokens_hook(step, x, logits)

        # Calculate statistics
        r_filled = (x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id).sum().item()
        a_filled = (x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id).sum().item()
        r_total = (cfg.r_span[1] - cfg.r_span[0]) * batch_size
        a_total = (cfg.a_span[1] - cfg.a_span[0]) * batch_size

        r_new = ((x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id) &
                 (x_old[:, cfg.r_span[0]:cfg.r_span[1]] == mask_token_id)).sum().item()
        a_new = ((x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id) &
                 (x_old[:, cfg.a_span[0]:cfg.a_span[1]] == mask_token_id)).sum().item()

        r_pct = 100 * r_filled / r_total
        a_pct = 100 * a_filled / a_total

        # Determine status
        if r_new > 0 and a_new > 0:
            status = "LOCKSTEP ✓"
        elif r_new > 0:
            status = "R only"
        elif a_new > 0:
            status = "A only"
        else:
            status = "waiting..."

        print(f"{step:4d} {r_pct:6.1f} {a_pct:6.1f} {r_new:6d} {a_new:6d} {status:>20}")

        output_history.append(x.clone())
        r_fill_history.append(r_pct)
        a_fill_history.append(a_pct)

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS:")
    print(f"  R block: {r_pct:.1f}% filled")
    print(f"  A block: {a_pct:.1f}% filled")
    print(f"  Fill ratio (A/R): {a_pct/max(r_pct, 0.1):.2f}")

    # Check for lockstep behavior
    lockstep_score = 1.0 - abs(r_pct - a_pct) / 100.0
    print(f"  Lockstep score: {lockstep_score:.2f} (1.0 = perfect lockstep)")

    if lockstep_score > 0.7:
        print("  ✅ GOOD LOCKSTEP: R and A are filling in tandem!")
    elif lockstep_score > 0.4:
        print("  ⚠️  PARTIAL LOCKSTEP: Some coupling between R and A")
    else:
        print("  ❌ POOR LOCKSTEP: R and A are filling independently")

    # Generate visualizations
    output_dir = Path(f"outputs/true_lockstep_{mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot fill progress
    ax = axes[0]
    steps = range(len(r_fill_history))
    ax.plot(steps, r_fill_history, 'r-', label='R Block', linewidth=2)
    ax.plot(steps, a_fill_history, 'b-', label='A Block', linewidth=2)
    ax.fill_between(steps, r_fill_history, a_fill_history, alpha=0.2, color='gray')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fill Percentage (%)')
    ax.set_title(f'Lockstep Progress ({mode})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot difference
    ax = axes[1]
    diff = [abs(r - a) for r, a in zip(r_fill_history, a_fill_history)]
    ax.plot(steps, diff, 'g-', linewidth=2)
    ax.fill_between(steps, 0, diff, alpha=0.3, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('|R% - A%| Difference')
    ax.set_title('Deviation from Perfect Lockstep')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
    ax.legend()

    plt.suptitle(f'True Lockstep Analysis - Score: {lockstep_score:.2f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_dir / "lockstep_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Analysis plot saved to: {plot_path}")

    # Create GIF
    if controller._commit_trace:
        gif_path = output_dir / "lockstep_decode.gif"
        create_decoding_gif(
            controller._commit_trace,
            output_history=output_history,
            r_span=cfg.r_span,
            a_span=cfg.a_span,
            tokenizer=tokenizer,
            output_path=str(gif_path),
            fps=3,
            max_display_len=seq_len
        )
        print(f"✓ GIF saved to: {gif_path}")

    return lockstep_score, r_pct, a_pct


def compare_modes():
    """Compare Gauss-Seidel vs Jacobi for lockstep behavior."""
    print("\n" + "="*70)
    print("COMPARING LOCKSTEP MODES")
    print("="*70)

    # Test Gauss-Seidel
    gs_score, gs_r, gs_a = test_true_lockstep("gauss_seidel")

    print("\n" + "-"*70 + "\n")

    # Test Jacobi
    j_score, j_r, j_a = test_true_lockstep("jacobi")

    print("\n" + "="*70)
    print("MODE COMPARISON SUMMARY:")
    print(f"  Gauss-Seidel: R={gs_r:.1f}%, A={gs_a:.1f}%, Score={gs_score:.2f}")
    print(f"  Jacobi:       R={j_r:.1f}%, A={j_a:.1f}%, Score={j_score:.2f}")

    if j_score > gs_score:
        print("\n✅ Jacobi mode achieves better lockstep behavior!")
    else:
        print("\n✅ Gauss-Seidel mode achieves better lockstep behavior!")


if __name__ == "__main__":
    compare_modes()
    print("\n🎉 True lockstep tests completed! Check 'outputs/true_lockstep_*' directories.")