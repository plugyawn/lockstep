"""
Dream-7B model loader and generation wrapper.
Handles model loading from HuggingFace and diffusion generation with hooks.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for Dream diffusion generation."""
    max_new_tokens: int = 512
    steps: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    alg: str = "entropy"  # confidence-based decoding algorithm
    alg_temp: float = 0.0
    output_history: bool = True
    return_dict_in_generate: bool = True


def load_dream(
    model_name: str = "Dream-org/Dream-v0-Instruct-7B",
    dtype: str = "bfloat16",
    device: str = "cuda"
) -> tuple:
    """
    Load Dream model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model path
        dtype: Model dtype (bfloat16, float16, float32)
        device: Device to load model on

    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading Dream model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Load model
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    # Move to device and set to eval mode
    model = model.to(device).eval()

    print(f"Model loaded successfully. Vocab size: {tokenizer.vocab_size}")

    return tokenizer, model


def dream_generate(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    controller=None,
    gen_cfg: Optional[GenerationConfig] = None
) -> Dict[str, Any]:
    """
    Generate text using Dream's diffusion generation with optional hooks.

    Args:
        model: Dream model instance
        tokenizer: Dream tokenizer
        input_ids: Input token IDs [B, L]
        attention_mask: Attention mask [B, L]
        controller: Optional LockstepController for R/A decoding
        gen_cfg: Generation configuration

    Returns:
        Generation output dictionary
    """
    if gen_cfg is None:
        gen_cfg = GenerationConfig()

    # Prepare kwargs for diffusion_generate
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": gen_cfg.max_new_tokens,
        "output_history": gen_cfg.output_history,
        "return_dict_in_generate": gen_cfg.return_dict_in_generate,
        "steps": gen_cfg.steps,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
        "alg": gen_cfg.alg,
        "alg_temp": gen_cfg.alg_temp,
    }

    # Add hooks if controller is provided
    if controller is not None:
        # Find block spans if not already set
        if controller.cfg.r_span is None or controller.cfg.a_span is None:
            controller.find_block_spans(input_ids)

        # Reset controller state
        controller.reset()

        # Add hook functions
        generate_kwargs["generation_logits_hook_func"] = controller.logits_hook
        generate_kwargs["generation_tokens_hook_func"] = controller.tokens_hook

    # Generate with Dream's diffusion method
    output = model.diffusion_generate(**generate_kwargs)

    return output


def prepare_ra_prompt(
    prompt: str,
    r_marker: str = "[R]",
    a_marker: str = "[A]"
) -> str:
    """
    Prepare a prompt with R/A block markers.

    Args:
        prompt: Base prompt text
        r_marker: Reasoning block marker
        a_marker: Answer block marker

    Returns:
        Formatted prompt with markers
    """
    # Simple format: prompt followed by R and A blocks
    formatted = f"{prompt}\n{r_marker} "

    # Add placeholder content for blocks (will be replaced during generation)
    formatted += "[reasoning placeholder] "
    formatted += f"{a_marker} [answer placeholder]"

    return formatted


def batch_generate(
    model,
    tokenizer,
    prompts: list,
    controller=None,
    gen_cfg: Optional[GenerationConfig] = None,
    batch_size: int = 4
) -> list:
    """
    Generate for multiple prompts in batches.

    Args:
        model: Dream model
        tokenizer: Dream tokenizer
        prompts: List of prompt strings
        controller: Optional LockstepController
        gen_cfg: Generation configuration
        batch_size: Batch size for generation

    Returns:
        List of generated texts
    """
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Move to model device
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)

        # Generate
        output = dream_generate(
            model,
            tokenizer,
            input_ids,
            attention_mask,
            controller,
            gen_cfg
        )

        # Decode outputs
        for j, seq in enumerate(output.sequences):
            # Remove input tokens
            generated_ids = seq[len(input_ids[j]):]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results.append(generated_text)

    return results