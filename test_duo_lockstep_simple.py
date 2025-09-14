#!/usr/bin/env python3
"""
Simplified DUO-130M lockstep test with simulated generation.
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SimpleLockstepVisualizer:
    """Simplified lockstep R/A visualization with simulated generation"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define R/A spans
        self.r_span = (15, 35)  # Reasoning block
        self.a_span = (35, 50)  # Answer block

        # Sample vocabulary for generation
        self.sample_words = [
            "understand", "analyze", "consider", "examine", "evaluate",
            "process", "identify", "recognize", "determine", "establish",
            "therefore", "thus", "hence", "consequently", "accordingly",
            "solution", "answer", "result", "conclusion", "outcome"
        ]

    def generate_lockstep(self, prompt, num_steps=8, max_length=64):
        """Generate text with simulated lockstep R/A control"""

        # Tokenize prompt
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
        states = []  # Track fill state at each step
        texts = []   # Track decoded text at each step
        fill_percentages = []  # Track R/A fill percentages

        # Create working text buffer
        tokens = self.tokenizer.decode(input_ids[0][:prompt_len], skip_special_tokens=True).split()

        # Pad with placeholders
        while len(tokens) < max_length:
            tokens.append("_")

        print(f"\nGenerating with {num_steps} steps...")
        print(f"Prompt ({prompt_len} tokens): {prompt}")
        print(f"R span: {self.r_span}, A span: {self.a_span}\n")

        # Initial state
        state = np.zeros(max_length)
        state[:prompt_len] = 1  # Mark prompt as filled
        states.append(state.copy())
        texts.append(" ".join(tokens))
        fill_percentages.append((0, 0))

        # Simulate generation over steps
        r_start, r_end = self.r_span
        a_start, a_end = self.a_span

        for step in range(num_steps):
            progress = (step + 1) / num_steps

            # Fill R block progressively
            if progress < 0.7:
                # Focus on R block
                r_fill_rate = min(0.3 + progress * 0.5, 1.0)
                num_r_fill = int((r_end - r_start) * r_fill_rate * progress)

                for i in range(min(num_r_fill, r_end - r_start)):
                    if tokens[r_start + i] == "_":
                        # Generate a reasoning word
                        word_idx = np.random.randint(0, 10)
                        tokens[r_start + i] = self.sample_words[word_idx]
                        state[r_start + i] = 2  # Mark as R block

            # Fill A block after R progresses
            if progress > 0.4:
                # Start filling A block
                a_fill_rate = min((progress - 0.4) * 2, 1.0)
                num_a_fill = int((a_end - a_start) * a_fill_rate)

                for i in range(min(num_a_fill, a_end - a_start)):
                    if tokens[a_start + i] == "_":
                        # Generate an answer word
                        word_idx = np.random.randint(10, 20)
                        tokens[a_start + i] = self.sample_words[word_idx]
                        state[a_start + i] = 3  # Mark as A block

            # Calculate fill percentages
            r_filled = np.sum(state[r_start:r_end] > 0) / (r_end - r_start)
            a_filled = np.sum(state[a_start:a_end] > 0) / (a_end - a_start)

            # Save state
            states.append(state.copy())
            current_text = " ".join([t if t != "_" else "" for t in tokens])
            texts.append(current_text)
            fill_percentages.append((r_filled, a_filled))

            print(f"Step {step+1}/{num_steps}: R={r_filled:.1%}, A={a_filled:.1%}")
            print(f"  Text: {current_text[:80]}...")

        return states, texts, fill_percentages

    def create_gif(self, states, texts, fill_percentages, output_path):
        """Create animated GIF showing lockstep generation"""

        print(f"\nCreating GIF visualization...")

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 2], hspace=0.3)

        # Token fill visualization
        ax1 = fig.add_subplot(gs[0])
        # Fill progress chart
        ax2 = fig.add_subplot(gs[1])
        # Text display
        ax3 = fig.add_subplot(gs[2])

        frames = []
        num_steps = len(states)
        seq_len = len(states[0])
        r_start, r_end = self.r_span
        a_start, a_end = self.a_span

        for step in range(num_steps):
            # Clear axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax3.axis('off')

            # Token fill visualization
            ax1.set_title(f'Step {step}/{num_steps-1} - DUO Lockstep R/A Generation',
                         fontsize=14, fontweight='bold')

            # Draw tokens
            colors = {0: '#E0E0E0', 1: '#2E7D32', 2: '#1976D2', 3: '#F57C00'}
            for i in range(seq_len):
                color = colors.get(int(states[step][i]), '#E0E0E0')
                ax1.add_patch(Rectangle((i, 0), 1, 1,
                                       facecolor=color,
                                       edgecolor='black', linewidth=0.5))

            # Add block boundaries
            ax1.axvline(r_start, color='blue', linestyle='--', alpha=0.5, linewidth=2)
            ax1.axvline(r_end, color='blue', linestyle='--', alpha=0.5, linewidth=2)
            ax1.axvline(a_start, color='orange', linestyle='--', alpha=0.5, linewidth=2)
            ax1.axvline(a_end, color='orange', linestyle='--', alpha=0.5, linewidth=2)

            # Labels
            ax1.text((r_start + r_end)/2, 1.2, 'Reasoning (R)',
                    ha='center', color='blue', fontweight='bold', fontsize=12)
            ax1.text((a_start + a_end)/2, 1.2, 'Answer (A)',
                    ha='center', color='orange', fontweight='bold', fontsize=12)

            ax1.set_xlim(0, seq_len)
            ax1.set_ylim(0, 1.5)
            ax1.set_xlabel('Token Position')
            ax1.set_yticks([])

            # Fill progress chart
            r_fills = [p[0] * 100 for p in fill_percentages[:step+1]]
            a_fills = [p[1] * 100 for p in fill_percentages[:step+1]]
            steps_x = list(range(step+1))

            ax2.plot(steps_x, r_fills, 'b-o', label='Reasoning (R)', linewidth=2)
            ax2.plot(steps_x, a_fills, 'r-o', label='Answer (A)', linewidth=2)
            ax2.set_xlim(0, num_steps-1)
            ax2.set_ylim(0, 105)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Fill %')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_title('Lockstep Fill Progress', fontsize=12)

            # Text display
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)

            # Title
            ax3.text(0.5, 0.95, 'Generated Text', fontsize=14, fontweight='bold',
                    ha='center', transform=ax3.transAxes)

            # Display text with color coding
            text = texts[step] if step < len(texts) else ""

            # Wrap text
            import textwrap
            wrapped = textwrap.fill(text, width=80)

            # Color code the text
            lines = wrapped.split('\n')
            y_pos = 0.7
            for line in lines:
                ax3.text(0.05, y_pos, line, fontsize=11,
                        transform=ax3.transAxes,
                        fontfamily='monospace')
                y_pos -= 0.08

            # Add statistics
            if step > 0:
                r_pct, a_pct = fill_percentages[step]
                stats_text = f"R: {r_pct:.0%} | A: {a_pct:.0%} | Step: {step}"
                ax3.text(0.95, 0.05, stats_text, fontsize=10,
                        transform=ax3.transAxes, ha='right', alpha=0.7,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Capture frame
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frame = Image.open(buf).copy()
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
    print("DUO-130M LOCKSTEP R/A GENERATION (SIMULATED)")
    print("=" * 80)

    # Create visualizer
    viz = SimpleLockstepVisualizer()

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
        states, texts, fill_percentages = viz.generate_lockstep(
            prompt,
            num_steps=12,  # More steps for better visualization
            max_length=64
        )

        # Create GIF
        gif_path = output_dir / f"duo_lockstep_sim_{i+1}.gif"
        viz.create_gif(states, texts, fill_percentages, gif_path)

        # Save final text
        final_text = texts[-1] if texts else ""
        text_path = output_dir / f"duo_lockstep_sim_{i+1}.txt"
        with open(text_path, 'w') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Steps: {len(states)}\n")
            f.write(f"Final R fill: {fill_percentages[-1][0]:.1%}\n")
            f.write(f"Final A fill: {fill_percentages[-1][1]:.1%}\n")
            f.write(f"Final text:\n{final_text}\n")

        print(f"✓ Saved text to {text_path}")

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to {output_dir}/")
    print("Files generated:")
    for f in sorted(output_dir.glob("duo_lockstep_sim_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()