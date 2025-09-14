#!/bin/bash
# Smoke test for Dream lockstep decoding

set -e  # Exit on error

# Activate virtual environment
source .venv/bin/activate

# Run smoke test with different configurations
echo "Running smoke test for Dream lockstep decoding..."

# Gauss-Seidel mode
python -m src.main \
    model.hf_name=Dream-org/Dream-v0-Instruct-7B \
    decode.mode=gauss_seidel \
    model.generation.steps=12 \
    model.generation.max_new_tokens=256 \
    experiment.name=smoke_gauss_seidel \
    wandb.enabled=false \
    +test_prompts='["Explain why the sky is blue [R] [A]", "Write a Python function to sort a list [R] [A]"]'

# Jacobi mode
python -m src.main \
    model.hf_name=Dream-org/Dream-v0-Instruct-7B \
    decode.mode=jacobi \
    model.generation.steps=12 \
    model.generation.max_new_tokens=256 \
    experiment.name=smoke_jacobi \
    wandb.enabled=false \
    +test_prompts='["What is 2+2? [R] [A]", "How do neural networks work? [R] [A]"]'

echo "Smoke tests completed successfully!"