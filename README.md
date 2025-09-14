# Dream-7B Lockstep R/A Decoding + RL Pipeline

This repository implements **lockstep Reasoning/Answer (R/A) block decoding** for Dream-7B diffusion language model, with hooks for RL training using VERL and verifiable rewards from EvalPlus.

## Key Features

- **Lockstep R/A Decoding**: Implements both Gauss-Seidel and Jacobi block decoding strategies
- **Dream-7B Integration**: Uses Dream's `generation_logits_hook_func` and `generation_tokens_hook_func` for fine-grained control
- **Confidence-Based Gating**: Adaptive token commitment based on per-token confidence thresholds
- **Visualization**: Generates GIFs showing block-wise token filling order
- **W&B Logging**: Comprehensive metrics tracking including tokens/sec, fill rates, and GPU utilization
- **Hydra Configs**: Modular configuration system for easy experimentation
- **RL-Ready**: Foundation for GRPO training with VERL and EvalPlus rewards

## Installation

### Prerequisites
- Python 3.10+
- CUDA 12.1+ with GPU (minimum 20GB VRAM for Dream-7B)
- 8× H200 GPUs for full RL training (optional)

### Setup

1. Clone the repository:
```bash
git clone <repo_url>
cd lockstep
```

2. Install uv and create environment:
```bash
pip install --user uv
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
```

Or use the locked environment:
```bash
uv pip sync
```

## Quick Start

### Basic Generation with Lockstep Decoding

```python
from src.core.dream_loader import load_dream, dream_generate, GenerationConfig
from src.core.lockstep_controller import LockstepController, LockstepCfg

# Load Dream model
tokenizer, model = load_dream("Dream-org/Dream-v0-Instruct-7B")

# Configure lockstep controller
cfg = LockstepCfg(
    mode="gauss_seidel",  # or "jacobi"
    tau_r=0.90,           # R block confidence threshold
    tau_a=0.92,           # A block confidence threshold
    max_fill_frac=0.4     # max fraction to fill per step
)
controller = LockstepController(cfg, tokenizer)

# Prepare prompt with R/A markers
prompt = "Explain why the sky is blue [R] [A]"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate with lockstep decoding
output = dream_generate(
    model, tokenizer,
    inputs.input_ids, inputs.attention_mask,
    controller=controller
)
```

### Run Smoke Tests

```bash
./scripts/smoke_decode.sh
```

This runs basic tests with both Gauss-Seidel and Jacobi modes.

### Profile Performance

```bash
./scripts/profile_infer.sh
```

Tests different step counts (5, 10, 20, 50) and logs metrics to W&B.

## Configuration

The system uses Hydra for configuration management. Key config files:

- `conf/config.yaml`: Main configuration
- `conf/model/dream.yaml`: Dream model settings
- `conf/decode/lockstep.yaml`: Lockstep decoding parameters
- `conf/wandb/default.yaml`: W&B logging settings
- `conf/hardware/h200_8x.yaml`: Multi-GPU settings
- `conf/rl/grpo.yaml`: RL training configuration

### Override configurations at runtime:

```bash
python -m src.main \
    decode.mode=jacobi \
    decode.tau_r=0.85 \
    model.generation.steps=20 \
    wandb.enabled=true
```

## Key Components

### 1. Lockstep Controller (`src/core/lockstep_controller.py`)

Implements the core R/A block decoding logic:
- **Gauss-Seidel**: Sequential R→A updates
- **Jacobi**: Parallel R and A updates
- Confidence-based token commitment
- Fill rate capping to avoid independence violations

### 2. Dream Loader (`src/core/dream_loader.py`)

Handles Dream model loading and generation:
- HuggingFace model loading
- Hook integration
- Batch generation support

### 3. GIF Visualization (`src/core/gif_viz.py`)

Creates animated visualizations:
- Token fill order heatmaps
- Block-wise statistics
- Step-by-step decoding GIFs

### 4. W&B Logging (`src/core/wandb_utils.py`)

Comprehensive experiment tracking:
- Generation metrics (tokens/sec, fill rates)
- System metrics (GPU/CPU utilization)
- Visualization artifacts

## Experiments

### Quality-Speed Trade-offs

Adjust diffusion steps to balance speed and quality:

```python
gen_cfg = GenerationConfig(
    steps=5,   # Faster but coarser
    # steps=50,  # Slower but higher quality
    temperature=0.1,
    alg="entropy"  # Confidence-based decoding
)
```

### Block Decoding Modes

Compare Gauss-Seidel vs Jacobi:

```bash
# Gauss-Seidel (sequential R→A)
python -m src.main decode.mode=gauss_seidel

# Jacobi (parallel R and A)
python -m src.main decode.mode=jacobi
```

### Confidence Thresholds

Tune per-block confidence thresholds:

```bash
python -m src.main \
    decode.tau_r=0.85 \  # Lower threshold for R block
    decode.tau_a=0.95    # Higher threshold for A block
```

## RL Training (Future Work)

The pipeline is designed for RL training with:
- **VERL**: For GRPO/PPO with FSDP/LoRA
- **EvalPlus**: For verifiable code rewards
- **8× H200**: For distributed training

Key files for RL extension:
- `src/rl/policy_dream.py`: Actor wrapper
- `src/rl/rewards_evalplus.py`: Reward computation
- `src/rl/train_grpo.py`: Training loop

## Research Papers

Key references implemented:
1. **Dream 7B** (arXiv:2508.15487): Diffusion language model with hooks
2. **LLaDA** (arXiv:2502.09992): Masked diffusion at scale
3. **Fast-dLLM** (arXiv:2505.22618): Block-wise KV caching
4. **D2F** (arXiv:2508.09192): Block-causal student models

## Troubleshooting

### CUDA Out of Memory
- Reduce `model.generation.max_new_tokens`
- Use smaller batch size
- Enable gradient checkpointing in configs

### Slow Generation
- Reduce `model.generation.steps`
- Increase `decode.max_fill_frac`
- Use Jacobi mode for parallelism

### W&B Issues
- Set `WANDB_API_KEY` environment variable
- Use `wandb.enabled=false` to disable logging

## Citation

If you use this code, please cite:

```bibtex
@article{ye2025dream,
  title={Dream 7B: Diffusion Large Language Models},
  author={Ye, Jiacheng and others},
  journal={arXiv preprint arXiv:2508.15487},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Dream-7B team for the model and hook architecture
- VERL team for the RL framework
- EvalPlus team for verification infrastructure