"""
Distributed training utilities for multi-GPU Dream training.
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from functools import partial
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: str = "env://"
) -> None:
    """
    Initialize distributed training environment.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Backend to use (nccl, gloo)
        init_method: Initialization method
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank
        )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    logger.info(f"Initialized distributed: rank={rank}, world_size={world_size}")


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def setup_model_parallel(
    model: torch.nn.Module,
    use_fsdp: bool = True,
    mixed_precision: Optional[str] = "bf16",
    cpu_offload: bool = False,
    transformer_layer_cls: Optional[type] = None
) -> torch.nn.Module:
    """
    Setup model for distributed training with FSDP or DDP.

    Args:
        model: Model to wrap
        use_fsdp: Whether to use FSDP (True) or DDP (False)
        mixed_precision: Mixed precision type (bf16, fp16, None)
        cpu_offload: Whether to offload to CPU (FSDP only)
        transformer_layer_cls: Transformer layer class for auto-wrapping

    Returns:
        Wrapped model
    """
    if not dist.is_initialized():
        logger.warning("Distributed not initialized, returning original model")
        return model

    device_id = torch.cuda.current_device()

    if use_fsdp:
        # Configure mixed precision
        mp_policy = None
        if mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif mixed_precision == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # Configure CPU offload
        cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

        # Configure auto-wrap policy for transformer layers
        wrap_policy = None
        if transformer_layer_cls:
            wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_cls},
            )

        # Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload_config,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=device_id,
            use_orig_params=True,  # Important for LoRA compatibility
        )

        logger.info("Model wrapped with FSDP")

    else:
        # Use standard DDP
        model = model.to(device_id)
        model = DDP(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False
        )

        logger.info("Model wrapped with DDP")

    return model


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce a tensor and compute mean across processes.

    Args:
        tensor: Tensor to reduce

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    if world_size == 1:
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather tensors from all processes.

    Args:
        tensor: Local tensor

    Returns:
        Concatenated tensor from all processes
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # Create list for gathered tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    # Concatenate
    return torch.cat(gathered, dim=0)


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from source rank to all ranks.

    Args:
        obj: Object to broadcast (only needed on src rank)
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj

    objects = [obj if get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


class DistributedSampler:
    """
    Simple distributed sampler for splitting data across processes.
    """

    def __init__(self, dataset_size: int, batch_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rank = get_rank()
        self.world_size = get_world_size()

        # Calculate samples per process
        self.samples_per_rank = dataset_size // self.world_size
        self.total_size = self.samples_per_rank * self.world_size

    def get_indices(self, epoch: int = 0) -> list:
        """Get indices for current rank."""
        # Create indices
        indices = list(range(self.dataset_size))

        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - self.dataset_size)]

        # Shuffle if needed
        if self.shuffle:
            # Use epoch as seed for reproducibility
            g = torch.Generator()
            g.manual_seed(epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # Subset for current rank
        indices = indices[self.rank:self.total_size:self.world_size]

        return indices


def setup_environment_variables(cfg: Dict[str, Any]) -> None:
    """
    Setup environment variables for optimal distributed training.

    Args:
        cfg: Configuration dictionary
    """
    env_vars = cfg.get('hardware', {}).get('env', {})

    for key, value in env_vars.items():
        os.environ[key] = str(value)

    # Additional optimizations
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere/Hopper GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set CUDA device flags
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;9.0'  # A100, A6000, H100/H200

    logger.info("Environment variables configured for distributed training")