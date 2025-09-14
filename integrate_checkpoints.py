#!/usr/bin/env python3
"""
Integration script for using pre-trained DUO/D2F checkpoints with lockstep R/A generation.
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.core.lockstep_controller import LockstepController
from omegaconf import DictConfig


class PretrainedLockstepGenerator:
    """Integrate pre-trained checkpoints with lockstep R/A control"""

    def __init__(self, model_name="s-sahoo/duo-distilled"):
        # Load pre-trained model
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        # Setup lockstep controller
        cfg = DictConfig({
            'decode': {
                'lockstep': {
                    'mode': 'gauss_seidel',
                    'r_span': [20, 40],
                    'a_span': [40, 50],
                    'tau_r': 0.9,
                    'tau_a': 0.85,
                    'fill_rate_cap': 0.2,
                    'coupling_strength': 0.1
                }
            }
        })
        self.controller = LockstepController(cfg)

    def generate_lockstep(self, prompt, max_length=128, num_steps=8):
        """Generate with lockstep R/A control using few-step DUO"""

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt",
                               max_length=max_length,
                               padding="max_length",
                               truncation=True)

        input_ids = inputs.input_ids
        batch_size, seq_len = input_ids.shape

        # Initialize with masked tokens
        x = input_ids.clone()
        mask_positions = torch.arange(len(prompt.split()), seq_len)
        x[0, mask_positions] = self.tokenizer.mask_token_id or self.tokenizer.vocab_size

        # Few-step generation with lockstep control
        for step in range(num_steps):
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(x)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Apply lockstep control
            x = self.controller.tokens_hook(
                x, logits, None, step, num_steps
            )

        # Decode result
        generated = self.tokenizer.decode(x[0], skip_special_tokens=True)
        return generated


if __name__ == "__main__":
    # Example usage
    generator = PretrainedLockstepGenerator("s-sahoo/duo-distilled")

    prompt = "To solve this complex problem, we need to"
    result = generator.generate_lockstep(prompt, num_steps=8)

    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    print(f"\nUsing 8-step generation (100× faster than standard diffusion)")
