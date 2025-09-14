"""
DUO Distillation for Dream-7B Lockstep R/A Generation

Implements Discrete Consistency Distillation from the DUO paper to accelerate
Dream-7B diffusion language model inference by two orders of magnitude.
"""

import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from src.core.dream_loader import DreamLoader


@dataclass
class DUOConfig:
    """Configuration for DUO distillation"""
    integral_cache_path: str = "cache/dream_integral_cache.pkl"
    gamma_min: float = -20.0
    gamma_max: float = 20.0
    vocab_size: int = 32000  # Dream-7B vocabulary
    update_teacher_every: int = 10000
    teacher_ema: bool = False
    linear_growth_dt: bool = False
    linear_growth_min: float = 0.01
    linear_growth_max: float = 0.1
    distillation_steps: int = 50000
    learning_rate: float = 6e-5
    batch_size: int = 32
    T: int = 512  # Total diffusion steps
    target_steps: int = 8  # Few-step generation target


class IntegralCache:
    """
    Computes and caches the integral for mapping Gaussian to discrete diffusion.
    Based on Eq. 10 from the DUO paper.
    """

    @staticmethod
    def compute(vocab_size: int, num_points: int = 10000) -> Dict[str, np.ndarray]:
        """Compute integral cache for given vocabulary size"""
        gamma_min, gamma_max = -20.0, 20.0
        gamma_range = np.linspace(gamma_min, gamma_max, num_points)

        # Compute p_t values (uniform-state diffusion probabilities)
        # This is the key mapping from Gaussian to discrete
        pt_values = []
        grad_pt_values = []

        for gamma in gamma_range:
            alpha = torch.sigmoid(torch.tensor(-gamma)).sqrt()
            pt = (alpha ** 2) / vocab_size + (1 - alpha ** 2)
            pt_values.append(pt.item())

            # Compute gradient for backprop
            grad_alpha = -0.5 * torch.sigmoid(torch.tensor(gamma)) * alpha
            grad_pt = 2 * alpha * grad_alpha * (1 - 1/vocab_size)
            grad_pt_values.append(grad_pt.item())

        return {
            'pt': np.array(pt_values),
            'grad_pt': np.array(grad_pt_values),
            'gamma_min': gamma_min,
            'gamma_max': gamma_max,
            'num_points': num_points,
            'vocab_size': vocab_size
        }


class DUODistillation(nn.Module):
    """
    DUO Distillation module for Dream-7B lockstep acceleration.
    Implements discrete consistency distillation to enable few-step generation.
    """

    def __init__(self, dream_model: nn.Module, config: DUOConfig):
        super().__init__()
        self.config = config

        # Teacher model (frozen Dream-7B)
        self.teacher = dream_model
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Student model (trainable copy)
        self.student = copy.deepcopy(dream_model)

        # Load or compute integral cache
        self._load_integral_cache()

        # Training state
        self.global_step = 0

    def _load_integral_cache(self):
        """Load or compute the integral cache"""
        cache_path = Path(self.config.integral_cache_path)

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.integral_cache = pickle.load(f)
        else:
            print(f"Computing integral cache for vocab_size={self.config.vocab_size}")
            self.integral_cache = IntegralCache.compute(self.config.vocab_size)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.integral_cache, f)
            print(f"Saved integral cache to {cache_path}")

    def _gamma_to_alphat(self, gamma_t: torch.Tensor) -> torch.Tensor:
        """Convert gamma to alpha_t using the integral cache"""
        gamma_min = self.integral_cache['gamma_min']
        gamma_max = self.integral_cache['gamma_max']
        num_points = self.integral_cache['num_points']

        # Clip gamma values
        gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)

        # Find indices in the cache
        indices = torch.round(
            (num_points - 1) * (gamma_t - gamma_min) / (gamma_max - gamma_min)
        ).long()

        # Look up pt values
        pt_tensor = torch.from_numpy(self.integral_cache['pt']).to(gamma_t.device)
        alpha_t = pt_tensor[indices]

        return alpha_t

    def _compute_dt(self) -> float:
        """Compute timestep delta for progressive distillation"""
        if self.config.linear_growth_dt:
            scale = self.global_step / self.config.distillation_steps
            return self.config.linear_growth_min + scale * (
                self.config.linear_growth_max - self.config.linear_growth_min
            )
        else:
            # Exponential growth: 2^n / T
            n = self.global_step // self.config.update_teacher_every
            return min(2 ** n / self.config.T, 0.5)

    def _sample_trajectory(
        self,
        x0: torch.Tensor,
        gamma_t: torch.Tensor,
        gamma_s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy states xt and xs from clean data x0"""
        x0_onehot = F.one_hot(x0, self.config.vocab_size).float()

        # Convert gamma to alpha/sigma
        alpha_t = torch.sigmoid(-gamma_t).sqrt().unsqueeze(-1).unsqueeze(-1)
        sigma_t = torch.sigmoid(gamma_t).sqrt().unsqueeze(-1).unsqueeze(-1)

        alpha_s = torch.sigmoid(-gamma_s).sqrt().unsqueeze(-1).unsqueeze(-1)
        sigma_s = torch.sigmoid(gamma_s).sqrt().unsqueeze(-1).unsqueeze(-1)

        # Sample noise
        epsilon = torch.randn_like(x0_onehot)

        # Forward diffusion
        xt = alpha_t * x0_onehot + sigma_t * epsilon
        xs = alpha_s * x0_onehot + sigma_s * epsilon

        # Convert to discrete tokens
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)

        return xt_discrete, xs_discrete

    def consistency_loss(
        self,
        x0: torch.Tensor,
        r_span: Tuple[int, int],
        a_span: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Compute consistency distillation loss for lockstep R/A blocks.

        Args:
            x0: Clean tokens [batch_size, seq_len]
            r_span: (start, end) indices for reasoning block
            a_span: (start, end) indices for answer block
        """
        batch_size = x0.shape[0]

        # Sample timesteps
        t = torch.rand(batch_size, device=x0.device)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        # Convert to gamma space
        gamma_t = self.config.gamma_min + t * (
            self.config.gamma_max - self.config.gamma_min
        )
        gamma_s = self.config.gamma_min + (t - dt) * (
            self.config.gamma_max - self.config.gamma_min
        )

        # Sample trajectories
        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)

        # Teacher prediction at time t
        with torch.no_grad():
            teacher_out_t = self.teacher(xt, gamma_t)
            teacher_probs_t = F.softmax(teacher_out_t, dim=-1)

        # Student predictions
        student_out_t = self.student(xt, gamma_t)
        student_out_s = self.student(xs, gamma_s)

        student_probs_t = F.softmax(student_out_t, dim=-1)
        student_probs_s = F.softmax(student_out_s, dim=-1)

        # Consistency loss: student at s should match student at t
        consistency = F.kl_div(
            student_probs_s.log(),
            student_probs_t.detach(),
            reduction='none'
        ).sum(-1)

        # Distillation loss: student at t should match teacher at t
        distillation = F.kl_div(
            student_probs_t.log(),
            teacher_probs_t,
            reduction='none'
        ).sum(-1)

        # Apply lockstep weighting (emphasize R/A blocks)
        weight = torch.ones_like(consistency)
        r_start, r_end = r_span
        a_start, a_end = a_span

        # Increase weight for R and A blocks
        weight[:, r_start:r_end] *= 2.0
        weight[:, a_start:a_end] *= 2.0

        # Combine losses
        loss = (consistency * weight).mean() + 0.1 * (distillation * weight).mean()

        return loss

    def generate_few_step(
        self,
        prompt: torch.Tensor,
        r_span: Tuple[int, int],
        a_span: Tuple[int, int],
        num_steps: int = 8,
        greedy_tail: bool = True,
        p_nucleus: float = 0.95
    ) -> torch.Tensor:
        """
        Few-step generation with lockstep R/A control.

        Args:
            prompt: Initial tokens [batch_size, prompt_len]
            r_span: (start, end) indices for reasoning block
            a_span: (start, end) indices for answer block
            num_steps: Number of denoising steps (default 8 vs 256 for regular)
            greedy_tail: Use greedy-tail sampling
            p_nucleus: Nucleus probability for greedy-tail
        """
        batch_size = prompt.shape[0]
        seq_len = prompt.shape[1]

        # Initialize with noise
        x = torch.randint(
            0, self.config.vocab_size,
            (batch_size, seq_len),
            device=prompt.device
        )

        # Keep prompt tokens fixed
        prompt_mask = prompt != -1
        x[prompt_mask] = prompt[prompt_mask]

        # Few-step denoising
        for step in range(num_steps):
            # Compute timestep
            t = 1.0 - (step + 1) / num_steps
            gamma = self.config.gamma_min + t * (
                self.config.gamma_max - self.config.gamma_min
            )

            # Denoise with student model
            with torch.no_grad():
                logits = self.student(x, gamma.expand(batch_size))
                probs = F.softmax(logits, dim=-1)

            # Apply lockstep coupling between R and A blocks
            r_start, r_end = r_span
            a_start, a_end = a_span

            # Boost A confidence based on R progress
            r_confidence = probs[:, r_start:r_end].max(-1)[0].mean(-1, keepdim=True)
            probs[:, a_start:a_end] = probs[:, a_start:a_end] * (1 + 0.1 * r_confidence.unsqueeze(-1))
            probs = probs / probs.sum(-1, keepdim=True)  # Renormalize

            # Greedy-tail sampling
            if greedy_tail:
                probs = self._apply_nucleus_filtering(probs, p_nucleus)

            # Sample next tokens
            x_next = torch.multinomial(probs.view(-1, self.config.vocab_size), 1)
            x_next = x_next.view(batch_size, seq_len)

            # Keep prompt and high-confidence tokens fixed
            confidence_threshold = 0.9
            high_conf_mask = probs.max(-1)[0] > confidence_threshold
            update_mask = ~prompt_mask & ~high_conf_mask
            x[update_mask] = x_next[update_mask]

        return x

    def _apply_nucleus_filtering(
        self,
        probs: torch.Tensor,
        p_nucleus: float
    ) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to probabilities"""
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find cutoff
        cutoff_mask = cumulative_probs > p_nucleus
        cutoff_mask[..., 1:] = cutoff_mask[..., :-1].clone()
        cutoff_mask[..., 0] = False

        # Zero out filtered probabilities
        sorted_probs[cutoff_mask] = 0.0

        # Restore original order
        probs_filtered = torch.zeros_like(probs)
        probs_filtered.scatter_(-1, sorted_indices, sorted_probs)

        # Renormalize
        probs_filtered = probs_filtered / probs_filtered.sum(-1, keepdim=True)

        return probs_filtered

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        r_span: Tuple[int, int],
        a_span: Tuple[int, int]
    ) -> Dict[str, float]:
        """Single training step for distillation"""
        x0 = batch['input_ids']

        # Compute loss
        loss = self.consistency_loss(x0, r_span, a_span)

        # Update global step
        self.global_step += 1

        # Maybe update teacher (if using EMA or periodic updates)
        if self.global_step % self.config.update_teacher_every == 0:
            if self.config.teacher_ema:
                # EMA update
                alpha = 0.999
                for teacher_param, student_param in zip(
                    self.teacher.parameters(),
                    self.student.parameters()
                ):
                    teacher_param.data = (
                        alpha * teacher_param.data +
                        (1 - alpha) * student_param.data
                    )
            else:
                # Hard update
                self.teacher.load_state_dict(self.student.state_dict())

        return {
            'loss': loss.item(),
            'dt': self._compute_dt(),
            'global_step': self.global_step
        }


def create_duo_lockstep(
    dream_model_path: str = "kuleshov-group/dream-7b",
    config: Optional[DUOConfig] = None
) -> DUODistillation:
    """
    Create a DUO distillation module for Dream-7B lockstep.

    Args:
        dream_model_path: Path to Dream-7B model
        config: DUO configuration

    Returns:
        DUODistillation module ready for training
    """
    if config is None:
        config = DUOConfig()

    # Load Dream-7B model
    dream_loader = DreamLoader()
    dream_model = dream_loader.load_model(dream_model_path)

    # Create distillation module
    duo = DUODistillation(dream_model, config)

    return duo