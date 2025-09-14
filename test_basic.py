#!/usr/bin/env python
"""
Basic test script for lockstep decoding without loading the full Dream model.
Uses a mock model for testing the controller logic.
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.core.lockstep_controller import LockstepController, LockstepCfg


def test_lockstep_controller():
    """Test the lockstep controller with mock data."""

    print("Testing Lockstep Controller...")

    # Mock configuration
    cfg = LockstepCfg(
        mode="gauss_seidel",
        tau_r=0.90,
        tau_a=0.92,
        max_fill_frac=0.4,
        halo=2,
        r_span=(10, 30),  # Mock R block span
        a_span=(30, 50),  # Mock A block span
    )

    controller = LockstepController(cfg)

    # Create mock data
    batch_size = 2
    seq_len = 60
    vocab_size = 100
    mask_token_id = 103  # BERT [MASK] token

    # Create mock sequence with some masked tokens
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    # Mask tokens in R and A regions
    x[:, 15:25] = mask_token_id  # Some R tokens
    x[:, 35:45] = mask_token_id  # Some A tokens

    # Create mock logits with varying confidence
    logits = torch.randn(batch_size, seq_len, vocab_size)
    # Make some positions more confident
    logits[:, 17:20, :] *= 3.0  # High confidence in R block
    logits[:, 37:40, :] *= 2.5  # Medium-high confidence in A block

    print(f"Input shape: {x.shape}")
    print(f"Masked positions in R block: {(x[:, cfg.r_span[0]:cfg.r_span[1]] == mask_token_id).sum().item()}")
    print(f"Masked positions in A block: {(x[:, cfg.a_span[0]:cfg.a_span[1]] == mask_token_id).sum().item()}")

    # Test logits hook
    state = {}
    processed_logits = controller.logits_hook(logits, state)
    assert processed_logits.shape == logits.shape
    assert controller._last_conf is not None
    print(f"✓ Logits hook processed successfully")

    # Test tokens hook (simulating multiple steps)
    num_steps = 5
    for step in range(num_steps):
        # Add some noise to logits to simulate evolution
        logits += torch.randn_like(logits) * 0.1

        # Process tokens
        x_new = controller.tokens_hook(step, x.clone(), logits)

        # Count commits
        commits = (x_new != x).sum().item()
        x = x_new

        print(f"  Step {step}: {commits} tokens committed")

    # Check commit trace
    assert len(controller._commit_trace) == num_steps
    print(f"✓ Tokens hook processed {num_steps} steps")

    # Test fill order grid
    fill_order = controller.get_fill_order_grid()
    assert fill_order.shape == (batch_size, seq_len)
    print(f"✓ Fill order grid shape: {fill_order.shape}")

    # Test Jacobi mode
    print("\nTesting Jacobi mode...")
    cfg.mode = "jacobi"
    controller_jacobi = LockstepController(cfg)

    # Reset mock data
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    x[:, 15:25] = mask_token_id
    x[:, 35:45] = mask_token_id

    for step in range(3):
        x = controller_jacobi.tokens_hook(step, x, logits)
        commits = (x != mask_token_id).sum().item() - (seq_len * batch_size - 20 * batch_size)
        print(f"  Step {step}: Total non-masked tokens: {(x != mask_token_id).sum().item()}")

    print("✓ All tests passed!")
    return True


if __name__ == "__main__":
    test_lockstep_controller()
    print("\nBasic controller tests completed successfully!")