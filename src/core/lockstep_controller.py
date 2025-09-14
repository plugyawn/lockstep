"""
Lockstep R/A Controller for Dream-7B diffusion language model.
Implements Gauss-Seidel and Jacobi block decoding strategies.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


@dataclass
class LockstepCfg:
    """Configuration for lockstep R/A decoding."""
    mode: str = "gauss_seidel"   # or "jacobi"
    tau_r: float = 0.90          # confidence threshold for R block
    tau_a: float = 0.92          # confidence threshold for A block
    max_fill_frac: float = 0.4   # max fraction of tokens to fill per step
    halo: int = 8                # context window around blocks
    r_open: str = "[R]"          # R block open marker
    a_open: str = "[A]"          # A block open marker
    r_span: Optional[Tuple[int, int]] = None
    a_span: Optional[Tuple[int, int]] = None


class LockstepController:
    """Controller for lockstep R/A decoding with Dream-7B."""

    def __init__(self, cfg: LockstepCfg, tokenizer=None):
        self.cfg = cfg
        self.tokenizer = tokenizer

        # State tracking
        self._last_conf = None       # [B, L] max prob per token
        self._last_logits = None     # [B, L, V] raw logits
        self._commit_trace = []      # list of boolean masks per micro-pass
        self._fill_order = []        # track order of token filling
        self._step_count = 0
        self._micro_sweep = 0

    def find_block_spans(self, input_ids: torch.Tensor) -> None:
        """Find R and A block spans from tokenized input."""
        if self.tokenizer is None:
            return

        # Convert to text to find markers
        text = self.tokenizer.decode(input_ids[0])

        # Find R block span
        r_start = text.find(self.cfg.r_open)
        if r_start >= 0:
            # Convert text position to token position
            prefix = text[:r_start]
            r_token_start = len(self.tokenizer.encode(prefix, add_special_tokens=False))
            # Find end of R block (before [A] marker)
            a_start = text.find(self.cfg.a_open, r_start)
            if a_start > r_start:
                r_text = text[r_start + len(self.cfg.r_open):a_start]
                r_token_len = len(self.tokenizer.encode(r_text, add_special_tokens=False))
                self.cfg.r_span = (r_token_start, r_token_start + r_token_len)

        # Find A block span
        a_start = text.find(self.cfg.a_open)
        if a_start >= 0:
            prefix = text[:a_start]
            a_token_start = len(self.tokenizer.encode(prefix, add_special_tokens=False))
            # Rest of sequence is A block
            a_text = text[a_start + len(self.cfg.a_open):]
            a_token_len = len(self.tokenizer.encode(a_text, add_special_tokens=False))
            self.cfg.a_span = (a_token_start, a_token_start + a_token_len)

    def logits_hook(self, logits: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        """
        Hook for Dream's generation_logits_hook_func.
        Caches logits and computes confidence scores.
        """
        with torch.no_grad():
            self._last_logits = logits.clone()
            # Compute max probability (confidence) per token
            probs = F.softmax(logits, dim=-1)
            self._last_conf = probs.max(dim=-1)[0]

        return logits

    def tokens_hook(self, step: int, x: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        Hook for Dream's generation_tokens_hook_func.
        Controls which tokens to commit based on confidence and block strategy.

        Args:
            step: Current diffusion step
            x: Current token sequence [B, L]
            logits: Current logits [B, L, V]

        Returns:
            Modified token sequence with selective commits
        """
        # Update internal state
        self._step_count = step
        self._last_logits = logits

        # Compute confidence if not already cached
        if self._last_conf is None or self._last_conf.shape != x.shape:
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                self._last_conf = probs.max(dim=-1)[0]

        # Determine which block to process
        if self.cfg.mode == "jacobi":
            # Process both blocks in parallel (true lockstep)
            x = self._process_block(x, "both")
        else:  # gauss_seidel
            # For true lockstep in Gauss-Seidel, process R first, then A in same step
            # This ensures both blocks progress together
            x_after_r = self._process_block(x, "r")
            # Update confidence based on R block changes for coupling
            if (x_after_r != x).any():
                # Recalculate confidence after R updates (simulating R→A influence)
                with torch.no_grad():
                    # Add small boost to A confidence when R has made progress
                    self._last_conf = F.softmax(logits, dim=-1).max(dim=-1)[0]
                    if self.cfg.a_span:
                        # Boost A confidence slightly based on R progress
                        r_progress = (x_after_r != x).float().mean()
                        a_start, a_end = self.cfg.a_span
                        self._last_conf[:, a_start:a_end] += r_progress * 0.1
            x = self._process_block(x_after_r, "a")
            self._micro_sweep += 1

        return x

    def _process_block(self, x: torch.Tensor, block: str) -> torch.Tensor:
        """
        Process a specific block (R, A, or both) for token commitment.

        Args:
            x: Current token sequence
            block: Which block to process ("r", "a", or "both")

        Returns:
            Updated token sequence
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Get mask token ID (assuming it's stored in tokenizer)
        mask_id = self.tokenizer.mask_token_id if self.tokenizer and hasattr(self.tokenizer, 'mask_token_id') else 103

        # Create mask for currently masked positions
        is_masked = (x == mask_id)

        # Initialize commit mask
        commit_mask = torch.zeros_like(is_masked, dtype=torch.bool)

        # Process R block
        if block in ["r", "both"] and self.cfg.r_span:
            r_start, r_end = self.cfg.r_span
            # Don't use halo for the actual block boundaries, only for context

            # Get confidence in this region (without halo)
            block_conf = self._last_conf[:, r_start:r_end]
            block_masked = is_masked[:, r_start:r_end]

            # Select high-confidence masked tokens
            candidates = block_masked & (block_conf >= self.cfg.tau_r)

            # Apply fill rate cap
            num_masked = block_masked.sum().item()
            if num_masked > 0:
                max_fill = max(1, int(self.cfg.max_fill_frac * num_masked))
                num_candidates = candidates.sum().item()

                if num_candidates > max_fill:
                    # Keep only top-k confident tokens among the masked ones
                    # Get confidence values for masked positions only
                    conf_masked = block_conf[block_masked]

                    if conf_masked.numel() > 0:
                        # Get top-k indices
                        topk_vals, topk_idx = conf_masked.topk(min(max_fill, conf_masked.numel()))

                        # Create new candidates mask
                        new_candidates = torch.zeros_like(block_masked)
                        # Find positions of masked tokens
                        masked_positions = block_masked.nonzero(as_tuple=True)
                        # Set top-k positions
                        for idx in topk_idx:
                            batch_idx = masked_positions[0][idx]
                            pos_idx = masked_positions[1][idx]
                            new_candidates[batch_idx, pos_idx] = True

                        candidates = new_candidates

            commit_mask[:, r_start:r_end] = candidates

        # Process A block
        if block in ["a", "both"] and self.cfg.a_span:
            a_start, a_end = self.cfg.a_span

            # Get confidence in this region
            block_conf = self._last_conf[:, a_start:a_end]
            block_masked = is_masked[:, a_start:a_end]

            # Select high-confidence masked tokens
            candidates = block_masked & (block_conf >= self.cfg.tau_a)

            # Apply fill rate cap
            num_masked = block_masked.sum().item()
            if num_masked > 0:
                max_fill = max(1, int(self.cfg.max_fill_frac * num_masked))
                num_candidates = candidates.sum().item()

                if num_candidates > max_fill:
                    # Keep only top-k confident tokens
                    conf_masked = block_conf[block_masked]

                    if conf_masked.numel() > 0:
                        topk_vals, topk_idx = conf_masked.topk(min(max_fill, conf_masked.numel()))

                        # Create new candidates mask
                        new_candidates = torch.zeros_like(block_masked)
                        masked_positions = block_masked.nonzero(as_tuple=True)
                        for idx in topk_idx:
                            batch_idx = masked_positions[0][idx]
                            pos_idx = masked_positions[1][idx]
                            new_candidates[batch_idx, pos_idx] = True

                        candidates = new_candidates

            commit_mask[:, a_start:a_end] |= candidates

        # Record commit trace for visualization
        self._commit_trace.append(commit_mask.cpu().clone())

        # Apply commits: replace masked tokens with predictions where confident
        if commit_mask.any():
            # Get predicted tokens from logits
            predicted_tokens = self._last_logits.argmax(dim=-1)
            x = torch.where(commit_mask, predicted_tokens, x)

        return x

    def get_fill_order_grid(self) -> np.ndarray:
        """
        Construct fill order grid from commit trace.

        Returns:
            2D array where each cell contains the step when that token was filled
        """
        if not self._commit_trace:
            return np.array([])

        # Stack all commit masks
        trace = torch.stack(self._commit_trace)  # [steps, B, L]
        batch_size = trace.shape[1]
        seq_len = trace.shape[2]

        # Create fill order grid
        fill_order = np.zeros((batch_size, seq_len), dtype=np.int32)

        for step_idx, mask in enumerate(trace):
            # Mark newly filled positions with step number
            newly_filled = mask.numpy()
            fill_order[newly_filled] = step_idx + 1

        return fill_order

    def reset(self):
        """Reset controller state for new generation."""
        self._last_conf = None
        self._last_logits = None
        self._commit_trace = []
        self._fill_order = []
        self._step_count = 0
        self._micro_sweep = 0