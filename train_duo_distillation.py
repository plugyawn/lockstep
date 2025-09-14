"""
Training script for DUO distillation of Dream-7B lockstep model.

This script implements discrete consistency distillation to accelerate
Dream-7B inference by two orders of magnitude (8 steps vs 256 steps).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from pathlib import Path

from src.core.duo_distillation import DUODistillation, DUOConfig, create_duo_lockstep
from src.core.dream_loader import DreamLoader


class DUOTrainer:
    """Trainer for DUO distillation"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.setup_model()

        # Setup data
        self.setup_data()

        # Setup optimizer
        self.setup_optimizer()

        # Setup logging
        self.setup_logging()

    def setup_model(self):
        """Initialize DUO distillation model"""
        print("Loading Dream-7B model for distillation...")

        # Create DUO config
        duo_config = DUOConfig(
            integral_cache_path=self.cfg.duo.integral_cache_path,
            distillation_steps=self.cfg.training.max_steps,
            learning_rate=self.cfg.optim.lr,
            batch_size=self.cfg.data.batch_size,
            T=self.cfg.duo.T,
            target_steps=self.cfg.duo.target_steps,
            update_teacher_every=self.cfg.duo.update_teacher_every,
            teacher_ema=self.cfg.duo.teacher_ema,
            linear_growth_dt=self.cfg.duo.linear_growth_dt
        )

        # Load base Dream model if finetuning from checkpoint
        if self.cfg.training.finetune_path:
            print(f"Loading checkpoint from {self.cfg.training.finetune_path}")
            dream_loader = DreamLoader()
            dream_model = dream_loader.load_model(self.cfg.training.finetune_path)
        else:
            dream_model = None

        # Create DUO distillation module
        self.model = create_duo_lockstep(
            dream_model_path=self.cfg.model.name_or_path,
            config=duo_config
        )
        self.model = self.model.to(self.device)

        # Define R/A spans based on config
        self.r_span = tuple(self.cfg.decode.lockstep.r_span)
        self.a_span = tuple(self.cfg.decode.lockstep.a_span)

        print(f"Model loaded. R span: {self.r_span}, A span: {self.a_span}")

    def setup_data(self):
        """Setup data loaders"""
        print("Loading dataset...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.tokenizer)

        # Load dataset
        if self.cfg.data.name == "openwebtext":
            dataset = load_dataset("openwebtext", split="train")
        else:
            dataset = load_dataset(self.cfg.data.name, split="train")

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.cfg.data.max_length,
                return_tensors="pt"
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.cfg.data.num_workers,
            remove_columns=dataset.column_names
        )

        # Create dataloader
        self.train_loader = DataLoader(
            tokenized_dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True
        )

        print(f"Dataset loaded: {len(tokenized_dataset)} samples")

    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.student.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay
        )

        # Linear warmup + cosine decay
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.optim.lr,
            total_steps=self.cfg.training.max_steps,
            pct_start=self.cfg.optim.warmup_ratio
        )

    def setup_logging(self):
        """Setup wandb logging"""
        if not self.cfg.wandb.offline:
            wandb.init(
                project=self.cfg.wandb.project,
                name=self.cfg.wandb.run_name,
                config=dict(self.cfg)
            )

    def train(self):
        """Main training loop"""
        print("Starting DUO distillation training...")

        global_step = 0
        progress_bar = tqdm(total=self.cfg.training.max_steps)

        for epoch in range(self.cfg.training.num_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                if global_step >= self.cfg.training.max_steps:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                self.optimizer.zero_grad()
                metrics = self.model.train_step(batch, self.r_span, self.a_span)
                loss = metrics['loss']

                # Backward pass
                loss_tensor = torch.tensor(loss, requires_grad=True, device=self.device)
                loss_tensor.backward()

                # Gradient clipping
                if self.cfg.optim.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.student.parameters(),
                        self.cfg.optim.grad_clip
                    )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()

                # Logging
                if global_step % self.cfg.logging.log_every == 0:
                    metrics['lr'] = self.scheduler.get_last_lr()[0]
                    if not self.cfg.wandb.offline:
                        wandb.log(metrics, step=global_step)

                    progress_bar.set_postfix({
                        'loss': f"{loss:.4f}",
                        'dt': f"{metrics['dt']:.4f}",
                        'lr': f"{metrics['lr']:.6f}"
                    })

                # Validation
                if global_step % self.cfg.training.val_every == 0:
                    self.validate(global_step)

                # Save checkpoint
                if global_step % self.cfg.training.save_every == 0:
                    self.save_checkpoint(global_step)

                global_step += 1
                progress_bar.update(1)

        progress_bar.close()
        print("Training completed!")

    @torch.no_grad()
    def validate(self, step: int):
        """Run validation and generate samples"""
        print(f"\nValidation at step {step}")

        # Generate samples with few-step generation
        prompt = "The key to understanding diffusion models is"
        prompt_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.data.max_length
        ).input_ids.to(self.device)

        # Generate with different step counts
        for num_steps in [4, 8, 16, 32]:
            generated = self.model.generate_few_step(
                prompt_tokens,
                self.r_span,
                self.a_span,
                num_steps=num_steps,
                greedy_tail=True
            )

            decoded = self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True
            )[0]

            print(f"[{num_steps} steps] {decoded[:200]}...")

            if not self.cfg.wandb.offline:
                wandb.log({
                    f"samples/{num_steps}_steps": decoded[:500]
                }, step=step)

    def save_checkpoint(self, step: int):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.cfg.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"duo_distilled_step_{step}.pt"

        torch.save({
            'step': step,
            'model_state_dict': self.model.student.state_dict(),
            'teacher_state_dict': self.model.teacher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': dict(self.cfg)
        }, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point"""

    # Override config for DUO distillation
    cfg.algo = "duo_distillation"
    cfg.training.max_steps = cfg.get('training', {}).get('max_steps', 50000)
    cfg.optim.lr = cfg.get('optim', {}).get('lr', 6e-5)

    # DUO specific settings
    cfg.duo = DictConfig({
        'integral_cache_path': 'cache/dream_integral_cache.pkl',
        'T': 512,
        'target_steps': 8,
        'update_teacher_every': 10000,
        'teacher_ema': False,
        'linear_growth_dt': False
    })

    # Create trainer
    trainer = DUOTrainer(cfg)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()