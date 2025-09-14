#!/bin/bash
# Profile inference performance for Dream lockstep decoding

set -e

source .venv/bin/activate

echo "Profiling Dream inference performance..."

# Run profiling with different step counts
for STEPS in 5 10 20 50; do
    echo "Testing with $STEPS diffusion steps..."

    python -m src.profile \
        model.generation.steps=$STEPS \
        model.generation.max_new_tokens=256 \
        decode.mode=gauss_seidel \
        decode.tau_r=0.90 \
        decode.tau_a=0.92 \
        hardware.profiling.enable_profiler=true \
        wandb.enabled=true \
        experiment.name="profile_steps_${STEPS}" \
        +num_iterations=10
done

echo "Profiling completed. Check W&B for detailed metrics."