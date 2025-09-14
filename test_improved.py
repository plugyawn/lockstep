#!/usr/bin/env python
"""
Improved test showing more realistic lockstep decoding behavior.
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
        """Convert token IDs to string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        text = ""
        for tid in token_ids:
            if tid == self.mask_token_id and not skip_special_tokens:
                text += "[M]"
            else:
                text += f"T{tid%100:02d} "
        return text.strip()


def simulate_realistic_decoding():
    """Simulate more realistic lockstep decoding with proper confidence evolution."""

    print("=" * 60)
    print("Realistic Lockstep R/A Decoding Simulation")
    print("=" * 60)

    # Setup
    batch_size = 2
    seq_len = 80
    vocab_size = 1000
    mask_token_id = 103

    # Create mock tokenizer
    tokenizer = MockTokenizer()

    # Configure lockstep controller with more aggressive thresholds
    cfg = LockstepCfg(
        mode="gauss_seidel",  # Sequential R -> A
        tau_r=0.70,  # Lower threshold for R (reasoning happens first)
        tau_a=0.75,  # Slightly higher for A (builds on R)
        max_fill_frac=0.25,  # Fill 25% of remaining per step
        halo=2,
        r_span=(20, 45),  # R block: positions 20-45 (25 tokens)
        a_span=(50, 70),  # A block: positions 50-70 (20 tokens)
    )

    controller = LockstepController(cfg, tokenizer)

    # Initialize sequence with masked tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Mask R and A regions completely
    x[:, cfg.r_span[0]:cfg.r_span[1]] = mask_token_id  # R block
    x[:, cfg.a_span[0]:cfg.a_span[1]] = mask_token_id  # A block

    print(f"Initial setup:")
    print(f"  Sequence shape: {x.shape}")
    print(f"  R block span: {cfg.r_span} ({cfg.r_span[1]-cfg.r_span[0]} tokens)")
    print(f"  A block span: {cfg.a_span} ({cfg.a_span[1]-cfg.a_span[0]} tokens)")
    print(f"  Mode: {cfg.mode}")
    print(f"  Thresholds: τ_R={cfg.tau_r}, τ_A={cfg.tau_a}")

    # Simulate diffusion steps
    num_steps = 20
    output_history = []

    # Track confidence evolution
    r_confidences = []
    a_confidences = []

    print(f"\nSimulating {num_steps} diffusion steps...")
    print(f"{'Step':>4} {'R New':>6} {'R Tot':>6} {'A New':>6} {'A Tot':>6} {'Total':>6}")
    print("-" * 40)

    for step in range(num_steps):
        # Generate mock logits with gradually increasing confidence
        base_noise = 1.5 - (step * 0.05)  # Decrease noise over time
        logits = torch.randn(batch_size, seq_len, vocab_size) * base_noise

        # Progressively increase confidence in R block (earlier steps)
        if step >= 1:
            r_start, r_end = cfg.r_span
            # Add confidence to random positions in R block
            r_confidence_boost = 2.0 + (step * 0.4)  # Increasing confidence
            num_confident = min(5 + step//2, (r_end - r_start)//2)

            for _ in range(num_confident):
                pos = torch.randint(r_start, r_end, (1,)).item()
                target_token = torch.randint(0, vocab_size, (batch_size,))
                for b in range(batch_size):
                    if x[b, pos] == mask_token_id:  # Only boost masked positions
                        logits[b, pos, target_token[b]] += r_confidence_boost

        # Add confidence to A block (later steps, after R has started filling)
        if step >= 4:
            a_start, a_end = cfg.a_span
            # A block confidence depends on R block progress
            r_filled = (x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id).float().mean()
            if r_filled > 0.2:  # Start filling A after 20% of R is done
                a_confidence_boost = 1.5 + (step * 0.3)
                num_confident = min(3 + step//3, (a_end - a_start)//2)

                for _ in range(num_confident):
                    pos = torch.randint(a_start, a_end, (1,)).item()
                    target_token = torch.randint(0, vocab_size, (batch_size,))
                    for b in range(batch_size):
                        if x[b, pos] == mask_token_id:
                            logits[b, pos, target_token[b]] += a_confidence_boost

        # Process through controller
        controller.logits_hook(logits, {})

        # Track mean confidence in each block
        if controller._last_conf is not None:
            r_conf = controller._last_conf[:, cfg.r_span[0]:cfg.r_span[1]].mean().item()
            a_conf = controller._last_conf[:, cfg.a_span[0]:cfg.a_span[1]].mean().item()
            r_confidences.append(r_conf)
            a_confidences.append(a_conf)

        # Apply tokens hook to update sequence
        x_old = x.clone()
        x = controller.tokens_hook(step, x, logits)

        # Count newly committed tokens per block
        r_new = ((x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id) &
                 (x_old[:, cfg.r_span[0]:cfg.r_span[1]] == mask_token_id)).sum().item()
        a_new = ((x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id) &
                 (x_old[:, cfg.a_span[0]:cfg.a_span[1]] == mask_token_id)).sum().item()

        r_total = (x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id).sum().item()
        a_total = (x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id).sum().item()
        total_filled = (x != mask_token_id).sum().item()

        # Store for history
        output_history.append(x.clone())

        print(f"{step:4d} {r_new:6d} {r_total:6d} {a_new:6d} {a_total:6d} {total_filled:6d}")

    # Calculate final statistics
    r_size = (cfg.r_span[1] - cfg.r_span[0]) * batch_size
    a_size = (cfg.a_span[1] - cfg.a_span[0]) * batch_size
    final_r_filled = (x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id).sum().item()
    final_a_filled = (x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id).sum().item()

    print("\n" + "=" * 40)
    print(f"Final statistics:")
    print(f"  R block: {final_r_filled}/{r_size} filled ({100*final_r_filled/r_size:.1f}%)")
    print(f"  A block: {final_a_filled}/{a_size} filled ({100*final_a_filled/a_size:.1f}%)")
    print(f"  Commit steps recorded: {len(controller._commit_trace)}")

    # Create output directory
    output_dir = Path("outputs/realistic_lockstep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print(f"\nGenerating visualizations...")

    # 1. Create GIF
    if controller._commit_trace:
        gif_path = output_dir / "realistic_decode.gif"
        create_decoding_gif(
            controller._commit_trace,
            output_history=output_history,
            r_span=cfg.r_span,
            a_span=cfg.a_span,
            tokenizer=tokenizer,
            output_path=str(gif_path),
            fps=3,  # Slightly faster
            max_display_len=seq_len
        )
        print(f"  ✓ GIF saved to: {gif_path}")

    # 2. Create fill order heatmap
    fill_order = controller.get_fill_order_grid()
    if fill_order.size > 0:
        fig = create_fill_order_heatmap(
            fill_order,
            cfg.r_span,
            cfg.a_span,
            tokenizer,
            x[0]
        )
        heatmap_path = output_dir / "fill_order_heatmap.png"
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Heatmap saved to: {heatmap_path}")

    # 3. Create statistics plot with confidence evolution
    if controller._commit_trace:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot confidence evolution
        ax = axes[0, 0]
        steps = range(len(r_confidences))
        ax.plot(steps, r_confidences, 'r-', label='R Block Mean Confidence', linewidth=2)
        ax.plot(steps, a_confidences, 'b-', label='A Block Mean Confidence', linewidth=2)
        ax.axhline(y=cfg.tau_r, color='r', linestyle='--', alpha=0.5, label=f'τ_R={cfg.tau_r}')
        ax.axhline(y=cfg.tau_a, color='b', linestyle='--', alpha=0.5, label=f'τ_A={cfg.tau_a}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Confidence')
        ax.set_title('Confidence Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Other plots from original statistics
        stats_path = output_dir / "enhanced_statistics.png"
        plt.suptitle(f'Lockstep {cfg.mode.replace("_", " ").title()} Decoding Analysis', fontsize=14)
        plt.tight_layout()
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Enhanced statistics saved to: {stats_path}")

    print(f"\n{'='*60}")
    print("Realistic simulation completed successfully!")
    print(f"Check outputs in: {output_dir}")

    return controller, output_history


if __name__ == "__main__":
    # Run realistic simulation
    controller, history = simulate_realistic_decoding()

    print("\n✅ Test completed! Check 'outputs/realistic_lockstep' for visualizations.")