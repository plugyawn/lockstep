"""
Weights & Biases logging utilities for Dream lockstep experiments.
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import time
import psutil
import GPUtil


class WandbLogger:
    """W&B logger for Dream lockstep experiments."""

    def __init__(self, cfg: Dict[str, Any], enabled: bool = True):
        self.cfg = cfg
        self.enabled = enabled and cfg.get('wandb', {}).get('enabled', True)
        self.run = None
        self.step = 0

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize W&B run."""
        wandb_cfg = self.cfg.get('wandb', {})

        # Initialize run
        self.run = wandb.init(
            project=wandb_cfg.get('project', 'dream-lockstep'),
            entity=wandb_cfg.get('entity'),
            name=self.cfg.get('experiment', {}).get('name', 'dream_run'),
            tags=wandb_cfg.get('tags', ['dream', 'lockstep']),
            config=self.cfg,
            reinit=True
        )

        # Define custom metrics
        wandb.define_metric("tokens_per_sec", summary="mean")
        wandb.define_metric("gpu_utilization", summary="mean")
        wandb.define_metric("memory_gb", summary="max")
        wandb.define_metric("fill_rate/*", step_metric="generation_step")
        wandb.define_metric("confidence/*", step_metric="generation_step")

    def log_generation_metrics(
        self,
        tokens_generated: int,
        time_elapsed: float,
        controller=None,
        step: Optional[int] = None
    ):
        """Log generation metrics."""
        if not self.enabled:
            return

        metrics = {
            'tokens_per_sec': tokens_generated / time_elapsed if time_elapsed > 0 else 0,
            'generation_time': time_elapsed,
            'tokens_generated': tokens_generated,
        }

        # Add controller metrics if available
        if controller and controller._commit_trace:
            # Calculate fill rates per block
            total_commits_r = 0
            total_commits_a = 0

            for mask in controller._commit_trace:
                mask_np = mask.numpy()
                if controller.cfg.r_span:
                    r_start, r_end = controller.cfg.r_span
                    total_commits_r += mask_np[:, r_start:r_end].sum()
                if controller.cfg.a_span:
                    a_start, a_end = controller.cfg.a_span
                    total_commits_a += mask_np[:, a_start:a_end].sum()

            if controller.cfg.r_span:
                r_size = controller.cfg.r_span[1] - controller.cfg.r_span[0]
                metrics['fill_rate/reasoning'] = total_commits_r / r_size if r_size > 0 else 0

            if controller.cfg.a_span:
                a_size = controller.cfg.a_span[1] - controller.cfg.a_span[0]
                metrics['fill_rate/answer'] = total_commits_a / a_size if a_size > 0 else 0

            # Add confidence metrics if available
            if controller._last_conf is not None:
                conf_np = controller._last_conf.cpu().numpy()
                metrics['confidence/mean'] = conf_np.mean()
                metrics['confidence/std'] = conf_np.std()

                if controller.cfg.r_span:
                    r_start, r_end = controller.cfg.r_span
                    r_conf = conf_np[:, r_start:r_end]
                    metrics['confidence/reasoning_mean'] = r_conf.mean()

                if controller.cfg.a_span:
                    a_start, a_end = controller.cfg.a_span
                    a_conf = conf_np[:, a_start:a_end]
                    metrics['confidence/answer_mean'] = a_conf.mean()

        # Add system metrics
        system_metrics = self.get_system_metrics()
        metrics.update(system_metrics)

        # Log to W&B
        wandb.log(metrics, step=step or self.step)
        self.step += 1

    def log_training_metrics(
        self,
        loss: float,
        rewards: Optional[List[float]] = None,
        learning_rate: Optional[float] = None,
        step: Optional[int] = None
    ):
        """Log training metrics."""
        if not self.enabled:
            return

        metrics = {
            'train/loss': loss,
        }

        if rewards:
            metrics['train/reward_mean'] = np.mean(rewards)
            metrics['train/reward_std'] = np.std(rewards)
            metrics['train/reward_max'] = np.max(rewards)
            metrics['train/reward_min'] = np.min(rewards)

        if learning_rate is not None:
            metrics['train/learning_rate'] = learning_rate

        # Add system metrics
        system_metrics = self.get_system_metrics()
        metrics.update(system_metrics)

        wandb.log(metrics, step=step or self.step)

    def log_eval_metrics(
        self,
        task: str,
        accuracy: float,
        additional_metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ):
        """Log evaluation metrics."""
        if not self.enabled:
            return

        metrics = {
            f'eval/{task}/accuracy': accuracy,
        }

        if additional_metrics:
            for key, value in additional_metrics.items():
                metrics[f'eval/{task}/{key}'] = value

        wandb.log(metrics, step=step or self.step)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str = "visualization",
        name: Optional[str] = None
    ):
        """Log artifact (e.g., GIF, model checkpoint)."""
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            name=name or f"{artifact_type}_{self.step}",
            type=artifact_type
        )
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)

    def log_histogram(
        self,
        data: np.ndarray,
        name: str,
        step: Optional[int] = None
    ):
        """Log histogram data."""
        if not self.enabled:
            return

        wandb.log({name: wandb.Histogram(data)}, step=step or self.step)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics (GPU, CPU, memory)."""
        metrics = {}

        # CPU metrics
        metrics['system/cpu_percent'] = psutil.cpu_percent()
        metrics['system/memory_percent'] = psutil.virtual_memory().percent
        metrics['system/memory_gb'] = psutil.virtual_memory().used / (1024 ** 3)

        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_utils = [gpu.load * 100 for gpu in gpus]
                gpu_mems = [gpu.memoryUtil * 100 for gpu in gpus]

                metrics['system/gpu_utilization'] = np.mean(gpu_utils)
                metrics['system/gpu_memory_percent'] = np.mean(gpu_mems)

                # Per-GPU metrics
                for i, (util, mem) in enumerate(zip(gpu_utils, gpu_mems)):
                    metrics[f'system/gpu_{i}_utilization'] = util
                    metrics[f'system/gpu_{i}_memory_percent'] = mem
        except:
            pass

        return metrics

    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """Watch model gradients and parameters."""
        if self.enabled:
            wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self):
        """Finish W&B run."""
        if self.enabled and self.run:
            self.run.finish()


def log_generation_sample(
    text: str,
    prompt: str,
    controller=None,
    step: Optional[int] = None
):
    """Log a generation sample to W&B."""
    if wandb.run is None:
        return

    # Create table
    columns = ["Step", "Prompt", "Generated", "Mode"]
    data = [[
        step or 0,
        prompt[:200],  # Truncate long prompts
        text[:500],    # Truncate long outputs
        controller.cfg.mode if controller else "N/A"
    ]]

    # Add block fill rates if available
    if controller and controller.cfg.r_span and controller.cfg.a_span:
        columns.extend(["R_Fill_Rate", "A_Fill_Rate"])

        # Calculate fill rates
        r_size = controller.cfg.r_span[1] - controller.cfg.r_span[0]
        a_size = controller.cfg.a_span[1] - controller.cfg.a_span[0]

        r_filled = 0
        a_filled = 0

        if controller._commit_trace:
            for mask in controller._commit_trace:
                mask_np = mask.numpy()
                r_filled += mask_np[:, controller.cfg.r_span[0]:controller.cfg.r_span[1]].sum()
                a_filled += mask_np[:, controller.cfg.a_span[0]:controller.cfg.a_span[1]].sum()

        data[0].extend([
            f"{100 * r_filled / r_size:.1f}%" if r_size > 0 else "N/A",
            f"{100 * a_filled / a_size:.1f}%" if a_size > 0 else "N/A"
        ])

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"generation_samples": table}, step=step)