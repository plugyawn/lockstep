"""
GIF visualization for Dream lockstep R/A decoding.
Creates animated visualizations showing the block-wise token filling process.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import imageio
from typing import List, Tuple, Optional, Dict
import torch
from PIL import Image
import io


def create_fill_order_heatmap(
    fill_order: np.ndarray,
    r_span: Optional[Tuple[int, int]] = None,
    a_span: Optional[Tuple[int, int]] = None,
    tokenizer=None,
    tokens=None,
    max_display_len: int = 100
) -> plt.Figure:
    """
    Create a heatmap visualization of the fill order.

    Args:
        fill_order: 2D array of fill order (step numbers)
        r_span: Tuple of (start, end) for R block
        a_span: Tuple of (start, end) for A block
        tokenizer: Optional tokenizer for token labels
        tokens: Optional token IDs for labels
        max_display_len: Maximum sequence length to display

    Returns:
        Matplotlib figure
    """
    # Truncate if too long
    if fill_order.shape[1] > max_display_len:
        fill_order = fill_order[:, :max_display_len]

    fig, ax = plt.subplots(figsize=(20, 6))

    # Create heatmap
    im = ax.imshow(fill_order, cmap='viridis', aspect='auto', interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fill Step', rotation=270, labelpad=15)

    # Add block overlays
    if r_span:
        r_start, r_end = r_span
        if r_end <= max_display_len:
            rect = patches.Rectangle(
                (r_start - 0.5, -0.5),
                r_end - r_start,
                fill_order.shape[0],
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                label='Reasoning (R)'
            )
            ax.add_patch(rect)

    if a_span:
        a_start, a_end = a_span
        if a_start < max_display_len:
            a_end = min(a_end, max_display_len)
            rect = patches.Rectangle(
                (a_start - 0.5, -0.5),
                a_end - a_start,
                fill_order.shape[0],
                linewidth=2,
                edgecolor='blue',
                facecolor='none',
                label='Answer (A)'
            )
            ax.add_patch(rect)

    # Labels
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Batch')
    ax.set_title('Lockstep R/A Token Fill Order')

    # Add legend if blocks are marked
    if r_span or a_span:
        ax.legend(loc='upper right')

    # Add token labels if available
    if tokenizer and tokens is not None:
        try:
            token_strs = [tokenizer.decode([t]) for t in tokens[0][:max_display_len]]
            # Only show every nth token to avoid crowding
            step = max(1, len(token_strs) // 50)
            tick_positions = list(range(0, len(token_strs), step))
            tick_labels = [token_strs[i][:10] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        except:
            pass

    plt.tight_layout()
    return fig


def create_decoding_gif(
    commit_trace: List[torch.Tensor],
    output_history: Optional[List[torch.Tensor]] = None,
    r_span: Optional[Tuple[int, int]] = None,
    a_span: Optional[Tuple[int, int]] = None,
    tokenizer=None,
    output_path: str = "decode_order.gif",
    fps: int = 2,
    max_display_len: int = 100
) -> str:
    """
    Create an animated GIF showing the decoding process step by step.

    Args:
        commit_trace: List of commit masks from controller
        output_history: Optional list of token sequences at each step
        r_span: R block span
        a_span: A block span
        tokenizer: Optional tokenizer for decoding
        output_path: Path to save GIF
        fps: Frames per second
        max_display_len: Maximum sequence length to display

    Returns:
        Path to saved GIF
    """
    if not commit_trace:
        return ""

    frames = []
    batch_size = commit_trace[0].shape[0]
    seq_len = min(commit_trace[0].shape[1], max_display_len)

    # Create cumulative fill grid
    fill_grid = np.zeros((batch_size, seq_len), dtype=np.float32)

    for step_idx, mask in enumerate(commit_trace):
        # Update fill grid
        mask_np = mask[:, :seq_len].numpy()
        fill_grid[mask_np] = step_idx + 1

        # Create frame
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

        # Top plot: Current state
        im1 = ax1.imshow(fill_grid, cmap='viridis', aspect='auto', vmin=0, vmax=len(commit_trace))
        ax1.set_title(f'Step {step_idx + 1}/{len(commit_trace)} - Cumulative Fill State')
        ax1.set_ylabel('Batch')

        # Add block overlays
        if r_span:
            r_start, r_end = r_span
            if r_end <= seq_len:
                rect = patches.Rectangle(
                    (r_start - 0.5, -0.5),
                    r_end - r_start,
                    batch_size,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax1.add_patch(rect)

        if a_span:
            a_start, a_end = a_span
            if a_start < seq_len:
                a_end = min(a_end, seq_len)
                rect = patches.Rectangle(
                    (a_start - 0.5, -0.5),
                    a_end - a_start,
                    batch_size,
                    linewidth=2,
                    edgecolor='blue',
                    facecolor='none'
                )
                ax1.add_patch(rect)

        # Bottom plot: Current step commits
        current_commits = mask_np.astype(float)
        im2 = ax2.imshow(current_commits, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_title(f'Tokens Committed This Step')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Batch')

        # Add text if we have decoded tokens
        if output_history and step_idx < len(output_history) and tokenizer:
            try:
                tokens = output_history[step_idx][0][:seq_len]
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                # Replace mask tokens with underscores for visibility
                text = text.replace(tokenizer.mask_token, '_')
                # Truncate for display
                if len(text) > 150:
                    text = text[:150] + "..."
                ax2.text(0.5, -0.15, text, transform=ax2.transAxes,
                        fontsize=8, ha='center', wrap=True)
            except:
                pass

        plt.tight_layout()

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        plt.close(fig)

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"GIF saved to {output_path}")

    return output_path


def visualize_block_statistics(
    controller,
    output_path: str = "block_stats.png"
) -> plt.Figure:
    """
    Create visualization of block-wise statistics.

    Args:
        controller: LockstepController with commit trace
        output_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    if not controller._commit_trace:
        return None

    # Calculate statistics per block
    r_commits = []
    a_commits = []

    for mask in controller._commit_trace:
        mask_np = mask.numpy()
        if controller.cfg.r_span:
            r_start, r_end = controller.cfg.r_span
            r_commits.append(mask_np[:, r_start:r_end].sum())
        if controller.cfg.a_span:
            a_start, a_end = controller.cfg.a_span
            a_commits.append(mask_np[:, a_start:a_end].sum())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Commits over time
    ax = axes[0, 0]
    steps = range(1, len(r_commits) + 1)
    if r_commits:
        ax.plot(steps, r_commits, 'r-', label='Reasoning (R)', linewidth=2)
    if a_commits:
        ax.plot(steps, a_commits, 'b-', label='Answer (A)', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tokens Committed')
    ax.set_title('Token Commits per Block Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cumulative commits
    ax = axes[0, 1]
    if r_commits:
        ax.plot(steps, np.cumsum(r_commits), 'r-', label='Reasoning (R)', linewidth=2)
    if a_commits:
        ax.plot(steps, np.cumsum(a_commits), 'b-', label='Answer (A)', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cumulative Tokens')
    ax.set_title('Cumulative Token Commits')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Fill rate
    ax = axes[1, 0]
    if controller.cfg.r_span and r_commits:
        r_size = controller.cfg.r_span[1] - controller.cfg.r_span[0]
        r_fill_rate = np.cumsum(r_commits) / r_size
        ax.plot(steps, r_fill_rate, 'r-', label='Reasoning (R)', linewidth=2)
    if controller.cfg.a_span and a_commits:
        a_size = controller.cfg.a_span[1] - controller.cfg.a_span[0]
        a_fill_rate = np.cumsum(a_commits) / a_size
        ax.plot(steps, a_fill_rate, 'b-', label='Answer (A)', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Fill Rate')
    ax.set_title('Block Fill Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Plot 4: Confidence distribution (if available)
    ax = axes[1, 1]
    if controller._last_conf is not None:
        conf_np = controller._last_conf.cpu().numpy()
        if controller.cfg.r_span:
            r_start, r_end = controller.cfg.r_span
            r_conf = conf_np[:, r_start:r_end].flatten()
            ax.hist(r_conf, bins=50, alpha=0.5, color='red', label='Reasoning (R)')
        if controller.cfg.a_span:
            a_start, a_end = controller.cfg.a_span
            a_conf = conf_np[:, a_start:a_end].flatten()
            ax.hist(a_conf, bins=50, alpha=0.5, color='blue', label='Answer (A)')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Final Confidence Distribution by Block')
        ax.legend()

    plt.suptitle(f'Lockstep Decoding Statistics - Mode: {controller.cfg.mode}', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Statistics saved to {output_path}")

    return fig