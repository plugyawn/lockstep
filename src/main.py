"""
Main entry point for Dream lockstep R/A decoding experiments.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import time
import os
from pathlib import Path

from src.core.dream_loader import load_dream, dream_generate, prepare_ra_prompt, GenerationConfig
from src.core.lockstep_controller import LockstepController, LockstepCfg
from src.core.gif_viz import create_decoding_gif, visualize_block_statistics, create_fill_order_heatmap
from src.core.wandb_utils import WandbLogger, log_generation_sample


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for Dream lockstep experiments."""

    print(OmegaConf.to_yaml(cfg))

    # Initialize W&B logger
    logger = WandbLogger(OmegaConf.to_container(cfg, resolve=True))

    # Load Dream model
    print(f"\nLoading Dream model: {cfg.model.hf_name}")
    tokenizer, model = load_dream(
        model_name=cfg.model.hf_name,
        dtype=cfg.model.dtype,
        device=cfg.model.device
    )

    # Create lockstep controller
    lockstep_cfg = LockstepCfg(
        mode=cfg.decode.mode,
        tau_r=cfg.decode.tau_r,
        tau_a=cfg.decode.tau_a,
        max_fill_frac=cfg.decode.max_fill_frac,
        halo=cfg.decode.halo,
        r_open=cfg.decode.block_markers.r_open,
        a_open=cfg.decode.block_markers.a_open
    )
    controller = LockstepController(lockstep_cfg, tokenizer)

    # Prepare generation config
    gen_cfg = GenerationConfig(
        max_new_tokens=cfg.model.generation.max_new_tokens,
        steps=cfg.model.generation.steps,
        temperature=cfg.model.generation.temperature,
        top_p=cfg.model.generation.top_p,
        alg=cfg.model.generation.alg,
        alg_temp=cfg.model.generation.alg_temp,
        output_history=cfg.model.generation.output_history,
        return_dict_in_generate=cfg.model.generation.return_dict_in_generate
    )

    # Test prompts
    test_prompts = cfg.get('test_prompts', [
        "Explain quantum computing in simple terms [R] [A]",
        "Write a Python function to calculate factorial [R] [A]",
        "What are the main causes of climate change? [R] [A]"
    ])

    print(f"\nRunning generation on {len(test_prompts)} prompts...")
    print(f"Mode: {cfg.decode.mode}")
    print(f"Steps: {cfg.model.generation.steps}")
    print(f"Max tokens: {cfg.model.generation.max_new_tokens}")

    # Process each prompt
    for idx, prompt in enumerate(test_prompts):
        print(f"\n{'='*60}")
        print(f"Prompt {idx+1}: {prompt[:100]}...")

        # Prepare input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=2048
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Reset controller
        controller.reset()

        # Generate with timing
        start_time = time.time()
        output = dream_generate(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            controller,
            gen_cfg
        )
        elapsed_time = time.time() - start_time

        # Decode output
        generated_ids = output.sequences[0][len(input_ids[0]):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\nGenerated ({elapsed_time:.2f}s):")
        print(generated_text[:500])

        # Log metrics
        tokens_generated = len(generated_ids)
        logger.log_generation_metrics(
            tokens_generated=tokens_generated,
            time_elapsed=elapsed_time,
            controller=controller,
            step=idx
        )

        # Log sample
        log_generation_sample(
            text=generated_text,
            prompt=prompt,
            controller=controller,
            step=idx
        )

        # Create visualizations
        if controller._commit_trace:
            output_dir = Path("outputs") / cfg.experiment.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create GIF
            gif_path = output_dir / f"decode_{idx}.gif"
            history = output.history if hasattr(output, 'history') else None
            create_decoding_gif(
                controller._commit_trace,
                output_history=history,
                r_span=controller.cfg.r_span,
                a_span=controller.cfg.a_span,
                tokenizer=tokenizer,
                output_path=str(gif_path),
                fps=2
            )

            # Log GIF to W&B
            if gif_path.exists():
                logger.log_artifact(str(gif_path), "visualization", f"decode_gif_{idx}")

            # Create statistics plot
            stats_path = output_dir / f"stats_{idx}.png"
            visualize_block_statistics(controller, str(stats_path))
            if stats_path.exists():
                logger.log_artifact(str(stats_path), "visualization", f"stats_{idx}")

            # Create fill order heatmap
            fill_order = controller.get_fill_order_grid()
            if fill_order.size > 0:
                fig = create_fill_order_heatmap(
                    fill_order,
                    controller.cfg.r_span,
                    controller.cfg.a_span,
                    tokenizer,
                    generated_ids
                )
                heatmap_path = output_dir / f"heatmap_{idx}.png"
                fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
                logger.log_artifact(str(heatmap_path), "visualization", f"heatmap_{idx}")

    # Finish
    logger.finish()
    print(f"\n{'='*60}")
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()