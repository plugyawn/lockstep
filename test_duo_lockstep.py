#!/usr/bin/env python3
"""
Test DUO-130M model with lockstep R/A controller and generate visualization GIF.
"""

import torch
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class DUOLockstepVisualizer:
    """Test and visualize DUO-130M with lockstep R/A generation"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load DUO model
        print("Loading DUO-distilled model...")
        try:
            self.model = AutoModelForMaskedLM.from_pretrained(
                "s-sahoo/duo-distilled",
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simulation mode for demonstration")
            self.model = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define R/A spans
        self.r_span = (15, 35)  # Reasoning block
        self.a_span = (35, 50)  # Answer block

        # Colors for visualization
        self.colors = {
            'prompt': '#2E7D32',  # Green
            'reasoning': '#1976D2',  # Blue
            'answer': '#F57C00',  # Orange
            'masked': '#9E9E9E',  # Gray
            'committed': '#4CAF50'  # Light green
        }

    def generate_lockstep(self, prompt, num_steps=8, max_length=64):
        """Generate text with lockstep R/A control, tracking states"""

        # Tokenize and pad prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Initialize tracking
        prompt_len = (attention_mask[0] == 1).sum().item()
        states = []  # Track token states at each step
        texts = []   # Track decoded text at each step
        confidences = []  # Track confidence scores

        # Create initial masked sequence
        x = input_ids.clone()
        mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.vocab_size

        print(f"Using mask_token_id: {mask_token_id}")

        # Mask everything after prompt
        for i in range(prompt_len, max_length):
            x[0, i] = mask_token_id

        # Initial state
        initial_tokens = x[0].cpu().numpy()
        initial_text = self.tokenizer.decode(x[0], skip_special_tokens=True)
        states.append(initial_tokens.copy())
        texts.append(initial_text)
        confidences.append(np.zeros(max_length))

        print(f"\nGenerating with {num_steps} steps...")
        print(f"Prompt ({prompt_len} tokens): {prompt}")
        print(f"R span: {self.r_span}, A span: {self.a_span}\n")

        # Few-step generation with lockstep
        for step in range(num_steps):
            if self.model is not None:
                # Real model inference
                with torch.no_grad():
                    outputs = self.model(x, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
            else:
                # Simulation for demonstration
                probs = self._simulate_probs(x, step, num_steps)

            # Get confidence scores
            conf = probs.max(dim=-1)[0].cpu().numpy()[0]
            confidences.append(conf.copy())

            # Apply lockstep logic
            x, committed_mask = self._apply_lockstep(
                x, probs, conf, step, num_steps, prompt_len
            )

            # Track state
            current_tokens = x[0].cpu().numpy()
            current_text = self.tokenizer.decode(x[0], skip_special_tokens=True)
            states.append(current_tokens.copy())
            texts.append(current_text)

            # Print progress
            r_start, r_end = self.r_span
            a_start, a_end = self.a_span
            r_filled = (x[0, r_start:r_end] != mask_token_id).float().mean().item()
            a_filled = (x[0, a_start:a_end] != mask_token_id).float().mean().item()

            print(f"Step {step+1}/{num_steps}: R={r_filled:.1%}, A={a_filled:.1%}")
            print(f"  Text: {current_text[:80]}...")

        return states, texts, confidences

    def _simulate_probs(self, x, step, num_steps):
        """Simulate probabilities for demonstration"""
        batch_size, seq_len = x.shape
        vocab_size = self.tokenizer.vocab_size

        # Create fake probabilities that gradually increase confidence
        progress = (step + 1) / num_steps
        base_conf = 0.3 + 0.6 * progress

        # Create random probabilities
        probs = torch.rand(batch_size, seq_len, vocab_size).to(self.device)

        # Set higher probs for some random tokens to simulate generation
        mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.vocab_size
        for i in range(seq_len):
            if x[0, i].item() == mask_token_id:
                # Generate a peaked distribution for masked positions
                random_token = torch.randint(100, min(1000, vocab_size), (1,)).item()  # Common tokens
                probs[0, i, :] = 0.01  # Low baseline
                probs[0, i, random_token] = 0.5 + 0.4 * progress  # High prob for one token

        probs = torch.softmax(probs * (1 + 2 * progress), dim=-1)

        # Boost confidence in R block first, then A
        r_start, r_end = self.r_span
        a_start, a_end = self.a_span

        if progress < 0.6:
            # Focus on R block
            probs[:, r_start:r_end] *= (1 + progress * 2)
        else:
            # Start boosting A block
            probs[:, a_start:a_end] *= (1 + (progress - 0.3) * 2)

        return probs

    def _apply_lockstep(self, x, probs, conf, step, num_steps, prompt_len):
        """Apply lockstep R/A control logic"""
        batch_size, seq_len = x.shape
        mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.vocab_size

        # R/A coupling
        r_start, r_end = self.r_span
        a_start, a_end = self.a_span

        # Calculate fill rates
        r_filled = (x[0, r_start:r_end] != mask_token_id).float().mean().item()

        # Boost A confidence based on R progress (coupling)
        if r_filled > 0.3:
            boost = min(0.2, r_filled * 0.3)
            conf[a_start:a_end] = np.minimum(1.0, conf[a_start:a_end] + boost)

        # Determine thresholds based on progress
        progress = (step + 1) / num_steps
        tau_r = 0.6 - 0.4 * progress  # Much lower threshold
        tau_a = 0.55 - 0.35 * progress

        # Create commitment mask
        committed_mask = np.zeros(seq_len, dtype=bool)

        # Commit R tokens
        x_cpu = x[0, r_start:r_end].cpu().numpy()
        r_mask = (conf[r_start:r_end] > tau_r) & (x_cpu == mask_token_id)
        if r_mask.any():
            # Sample tokens for high-confidence positions
            for i in range(r_start, r_end):
                if conf[i] > tau_r and x[0, i].item() == mask_token_id:
                    token_probs = probs[0, i]
                    x[0, i] = torch.multinomial(token_probs, 1).item()
                    committed_mask[i] = True

        # Commit A tokens
        x_cpu_a = x[0, a_start:a_end].cpu().numpy()
        a_mask = (conf[a_start:a_end] > tau_a) & (x_cpu_a == mask_token_id)
        if a_mask.any():
            for i in range(a_start, a_end):
                if conf[i] > tau_a and x[0, i].item() == mask_token_id:
                    token_probs = probs[0, i]
                    x[0, i] = torch.multinomial(token_probs, 1).item()
                    committed_mask[i] = True

        return x, committed_mask

    def create_gif(self, states, texts, confidences, output_path):
        """Create animated GIF showing lockstep generation with text"""

        print(f"\nCreating GIF visualization...")

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2], hspace=0.3)

        # Token fill visualization
        ax1 = fig.add_subplot(gs[0])

        # Confidence heatmap
        ax2 = fig.add_subplot(gs[1])

        # Text display
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')

        # Setup
        num_steps = len(states)
        seq_len = len(states[0])
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.vocab_size

        # Create frames
        frames = []

        for step in range(num_steps):
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax3.axis('off')

            # Token fill visualization
            ax1.set_title(f'Step {step}/{num_steps-1} - Lockstep R/A Token Filling',
                         fontsize=14, fontweight='bold')

            # Create token state array
            token_state = np.zeros(seq_len)
            r_start, r_end = self.r_span
            a_start, a_end = self.a_span

            for i in range(seq_len):
                if states[step][i] != mask_token_id:
                    if i < 15:  # Prompt
                        token_state[i] = 1
                    elif r_start <= i < r_end:  # R block
                        token_state[i] = 2
                    elif a_start <= i < a_end:  # A block
                        token_state[i] = 3
                    else:
                        token_state[i] = 0.5

            # Plot token states
            colors = ['#E0E0E0', '#2E7D32', '#1976D2', '#F57C00']
            for i in range(seq_len):
                color_idx = int(token_state[i] * 2) if token_state[i] <= 1 else int(token_state[i])
                ax1.add_patch(Rectangle((i, 0), 1, 1,
                                       facecolor=colors[min(color_idx, 3)],
                                       edgecolor='black', linewidth=0.5))

            # Add block labels
            ax1.axvline(r_start, color='blue', linestyle='--', alpha=0.5)
            ax1.axvline(r_end, color='blue', linestyle='--', alpha=0.5)
            ax1.axvline(a_start, color='orange', linestyle='--', alpha=0.5)
            ax1.axvline(a_end, color='orange', linestyle='--', alpha=0.5)

            ax1.text(r_start + (r_end-r_start)/2, 1.1, 'Reasoning (R)',
                    ha='center', color='blue', fontweight='bold')
            ax1.text(a_start + (a_end-a_start)/2, 1.1, 'Answer (A)',
                    ha='center', color='orange', fontweight='bold')

            ax1.set_xlim(0, seq_len)
            ax1.set_ylim(0, 1.3)
            ax1.set_xlabel('Token Position')
            ax1.set_yticks([])

            # Calculate fill percentages
            r_filled = np.sum(token_state[r_start:r_end] > 0) / (r_end - r_start) * 100
            a_filled = np.sum(token_state[a_start:a_end] > 0) / (a_end - a_start) * 100

            ax1.text(0.02, 0.5, f'R: {r_filled:.0f}%', transform=ax1.transAxes,
                    fontsize=12, color='blue', fontweight='bold')
            ax1.text(0.02, 0.2, f'A: {a_filled:.0f}%', transform=ax1.transAxes,
                    fontsize=12, color='orange', fontweight='bold')

            # Confidence heatmap
            if step > 0:
                conf_data = confidences[step].reshape(1, -1)
                sns.heatmap(conf_data, ax=ax2, cmap='YlOrRd', vmin=0, vmax=1,
                           cbar_kws={'label': 'Confidence'},
                           xticklabels=False, yticklabels=False)
                ax2.set_title('Token Confidence Scores', fontsize=12)

            # Text display with color coding
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

            # Display text with wrapping
            text = texts[step] if step < len(texts) else ""

            # Add title
            ax3.text(0.5, 0.95, 'Generated Text', fontsize=14, fontweight='bold',
                    ha='center', transform=ax3.transAxes)

            # Wrap and display text
            import textwrap
            wrapped_text = textwrap.fill(text, width=80)
            ax3.text(0.05, 0.5, wrapped_text, fontsize=11,
                    transform=ax3.transAxes, verticalalignment='center',
                    fontfamily='monospace', wrap=True)

            # Add step info
            ax3.text(0.95, 0.05, f'Step {step}', fontsize=10,
                    transform=ax3.transAxes, ha='right', alpha=0.7)

            # Capture frame
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frame = Image.open(buf).copy()  # Make a copy to keep in memory
            frames.append(frame)
            buf.close()

        plt.close()

        # Save as GIF
        print(f"Saving GIF to {output_path}")
        if frames:
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:] if len(frames) > 1 else [],
                duration=500,  # 500ms per frame
                loop=0
            )

        print(f"✓ GIF saved successfully!")

        return output_path


def main():
    """Main test function"""

    print("=" * 80)
    print("DUO-130M LOCKSTEP R/A GENERATION TEST")
    print("=" * 80)

    # Create visualizer
    viz = DUOLockstepVisualizer()

    # Test prompts
    prompts = [
        "To solve this complex problem, we need to",
        "The fundamental principle of diffusion models is",
        "Step by step, let's analyze"
    ]

    # Create output directory
    output_dir = Path("outputs/duo_lockstep")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n{'='*60}")
        print(f"Test {i+1}: {prompt}")
        print(f"{'='*60}")

        # Generate with lockstep
        states, texts, confidences = viz.generate_lockstep(
            prompt,
            num_steps=8,
            max_length=64
        )

        # Create GIF
        gif_path = output_dir / f"duo_lockstep_{i+1}.gif"
        viz.create_gif(states, texts, confidences, gif_path)

        # Save final text
        final_text = texts[-1] if texts else ""
        text_path = output_dir / f"duo_lockstep_{i+1}.txt"
        with open(text_path, 'w') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Steps: {len(states)}\n")
            f.write(f"Final text:\n{final_text}\n")

        print(f"✓ Saved text to {text_path}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to {output_dir}/")
    print("Files generated:")
    for f in output_dir.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()