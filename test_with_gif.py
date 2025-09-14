#!/usr/bin/env python
"""
Test lockstep controller with GIF generation using mock data.
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
                text += "[MASK]"
            else:
                text += f"T{tid} "
        return text.strip()


def simulate_lockstep_decoding():
    """Simulate lockstep decoding with visualization."""

    print("=" * 60)
    print("Testing Lockstep Decoding with Visualization")
    print("=" * 60)

    # Setup
    batch_size = 2
    seq_len = 80
    vocab_size = 1000
    mask_token_id = 103

    # Create mock tokenizer
    tokenizer = MockTokenizer()

    # Configure lockstep controller
    cfg = LockstepCfg(
        mode="gauss_seidel",
        tau_r=0.85,  # Lower threshold for testing
        tau_a=0.88,
        max_fill_frac=0.3,
        halo=4,
        r_span=(20, 40),  # R block: positions 20-40
        a_span=(45, 65),  # A block: positions 45-65
    )

    controller = LockstepController(cfg, tokenizer)

    # Initialize sequence with masked tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Mask R and A regions
    x[:, 20:40] = mask_token_id  # R block
    x[:, 45:65] = mask_token_id  # A block

    print(f"Initial setup:")
    print(f"  Sequence shape: {x.shape}")
    print(f"  R block span: {cfg.r_span}")
    print(f"  A block span: {cfg.a_span}")
    print(f"  Masked positions in R: {(x[:, cfg.r_span[0]:cfg.r_span[1]] == mask_token_id).sum().item()}")
    print(f"  Masked positions in A: {(x[:, cfg.a_span[0]:cfg.a_span[1]] == mask_token_id).sum().item()}")

    # Simulate diffusion steps
    num_steps = 15
    output_history = []

    print(f"\nSimulating {num_steps} diffusion steps...")

    for step in range(num_steps):
        # Generate mock logits with varying confidence
        logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0

        # Make some positions progressively more confident
        if step > 2:
            # Increase confidence in some R positions
            r_start, r_end = cfg.r_span
            confident_r = torch.randint(r_start, r_start + 10, (5,))
            for pos in confident_r:
                if pos < r_end:
                    # Create peaked distribution
                    target_token = torch.randint(0, vocab_size, (batch_size,))
                    for b in range(batch_size):
                        logits[b, pos, target_token[b]] += 4.0 + step * 0.3

        if step > 5:
            # Increase confidence in some A positions
            a_start, a_end = cfg.a_span
            confident_a = torch.randint(a_start, a_start + 10, (5,))
            for pos in confident_a:
                if pos < a_end:
                    target_token = torch.randint(0, vocab_size, (batch_size,))
                    for b in range(batch_size):
                        logits[b, pos, target_token[b]] += 3.5 + step * 0.2

        # Process through controller
        controller.logits_hook(logits, {})

        # Apply tokens hook to update sequence
        x_old = x.clone()
        x = controller.tokens_hook(step, x, logits)

        # Count newly committed tokens
        newly_committed = ((x != mask_token_id) & (x_old == mask_token_id)).sum().item()
        total_filled = (x != mask_token_id).sum().item()

        # Store for history
        output_history.append(x.clone())

        print(f"  Step {step:2d}: {newly_committed:3d} new commits, {total_filled:3d} total filled")

    # Calculate final statistics
    final_r_filled = (x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id).sum().item()
    final_a_filled = (x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id).sum().item()
    r_total = (cfg.r_span[1] - cfg.r_span[0]) * batch_size
    a_total = (cfg.a_span[1] - cfg.a_span[0]) * batch_size

    print(f"\nFinal statistics:")
    print(f"  R block: {final_r_filled}/{r_total} filled ({100*final_r_filled/r_total:.1f}%)")
    print(f"  A block: {final_a_filled}/{a_total} filled ({100*final_a_filled/a_total:.1f}%)")
    print(f"  Commit trace length: {len(controller._commit_trace)}")

    # Create output directory
    output_dir = Path("outputs/test_lockstep")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print(f"\nGenerating visualizations...")

    # 1. Create GIF
    if controller._commit_trace:
        gif_path = output_dir / "lockstep_decode.gif"
        create_decoding_gif(
            controller._commit_trace,
            output_history=output_history,
            r_span=cfg.r_span,
            a_span=cfg.a_span,
            tokenizer=tokenizer,
            output_path=str(gif_path),
            fps=2,
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

    # 3. Create statistics plot
    stats_path = output_dir / "block_statistics.png"
    fig = visualize_block_statistics(controller, str(stats_path))
    if fig:
        plt.close(fig)
        print(f"  ✓ Statistics saved to: {stats_path}")

    print(f"\n{'='*60}")
    print("Test completed successfully!")
    print(f"Check outputs in: {output_dir}")

    return controller, output_history


def test_jacobi_mode():
    """Test Jacobi mode specifically."""
    print("\n" + "="*60)
    print("Testing Jacobi Mode")
    print("="*60)

    batch_size = 1
    seq_len = 60
    vocab_size = 500
    mask_token_id = 103

    tokenizer = MockTokenizer()

    cfg = LockstepCfg(
        mode="jacobi",  # Parallel mode
        tau_r=0.80,
        tau_a=0.82,
        max_fill_frac=0.4,
        halo=2,
        r_span=(10, 25),
        a_span=(30, 45),
    )

    controller = LockstepController(cfg, tokenizer)

    # Initialize
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    x[:, 10:25] = mask_token_id  # R block
    x[:, 30:45] = mask_token_id  # A block

    print(f"Initial masked: R={15*batch_size}, A={15*batch_size}")

    # Run a few steps
    for step in range(5):
        logits = torch.randn(batch_size, seq_len, vocab_size) * 3.0

        # Add strong signals to both blocks
        for pos in range(12, 18):  # Some R positions
            logits[:, pos, pos % vocab_size] += 5.0
        for pos in range(32, 38):  # Some A positions
            logits[:, pos, (pos*2) % vocab_size] += 5.0

        controller.logits_hook(logits, {})
        x = controller.tokens_hook(step, x, logits)

        r_filled = (x[:, cfg.r_span[0]:cfg.r_span[1]] != mask_token_id).sum().item()
        a_filled = (x[:, cfg.a_span[0]:cfg.a_span[1]] != mask_token_id).sum().item()

        print(f"  Step {step}: R filled={r_filled}/15, A filled={a_filled}/15")

    print("Jacobi mode test completed!")


if __name__ == "__main__":
    # Run main test
    controller, history = simulate_lockstep_decoding()

    # Run Jacobi test
    test_jacobi_mode()

    print("\n✅ All tests completed! Check the 'outputs/test_lockstep' directory for visualizations.")