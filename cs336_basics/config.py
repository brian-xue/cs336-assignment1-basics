from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DatasetConfig:
    """
    Configuration for the dataset.
    """
    train_path: str = field(default="data/train.bin", metadata={"help": "Path to the training dataset."})
    val_path: str = field(default="data/val.bin", metadata={"help": "Path to the validation dataset."})
    # data type
    np_dtype: str = field(default="uint16", metadata={"help": "Data type of the dataset. Options: 'uint8', 'uint16', 'uint32', 'uint64'."})

    context_length: int = field(default=128, metadata={"help": "Length of the context window for training."})

    device: str = field(default="cpu", metadata={"help": "Device to use for training. Options: 'cpu', 'cuda', 'cuda:<device_id>'."})

@dataclass
class ModelConfig:
    """
    Configuration for the model.
    """
    vocab_size: int = field(default=10000, metadata={"help": "Size of the vocabulary for the model."})
    context_length: int = field(default=128, metadata={"help": "Length of the context window for the model."})

    d_model: int = field(default=256, metadata={"help": "Dimensionality of the model's hidden states."})
    num_layers: int = field(default=4, metadata={"help": "Number of layers in the model."})
    num_heads: int = field(default=8, metadata={"help": "Number of attention heads in the model."})

    d_off: Optional[int] = field(default=None, metadata={"help": "Dimensionality of the feedforward network. If None, defaults to 4 * d_model."})
    rope_theta: float = field(default=10000.0, metadata={"help": "Theta value for Rotary Positional Embeddings (RoPE)."})
    max_seq_len: Optional[int] = field(default=None, metadata={"help": "Maximum sequence length for the model. If None, defaults to context_length."})

    rmsnorm_eps: float = field(default=1e-5, metadata={"help": "Epsilon value for RMSNorm."})
    torch_dtype: str = field(default="float32", metadata={"help": "Data type for the model's parameters. Options: 'float32', 'float16', 'bfloat16'."})

@dataclass
class OptimizerConfig:
    """
    Configuration for the optimizer.
    """
    lr_max: float = field(default=1e-3, metadata={"help": "Maximum learning rate for the optimizer."})
    lr_min: float = field(default=1e-5, metadata={"help": "Minimum learning rate for the optimizer."})
    warmup_iters: int = field(default=2000, metadata={"help": "Number of warmup iterations for the learning rate scheduler."})
    cosine_cycle_iters: int = field(default=10000, metadata={"help": "Number of iterations for a full cosine cycle in the learning rate scheduler."})
    
    beta1: float = field(default=0.9, metadata={"help": "Beta1 parameter for the Adam optimizer."})
    beta2: float = field(default=0.95, metadata={"help": "Beta2 parameter for the Adam optimizer."})
    eps: float = field(default=1e-8, metadata={"help": "Epsilon value for the Adam optimizer."})
    weight_decay: float = field(default=0.1, metadata={"help": "Weight decay for the Adam optimizer."})

    grad_clip: float = field(default=1.0, metadata={"help": "Gradient clipping value for the optimizer."})

@dataclass
class TrainingConfig:
    """
    Configuration for the training process.
    """
    num_iters: int = field(default=10000, metadata={"help": "Total number of training iterations."})
    batch_size: int = field(default=64, metadata={"help": "Batch size for training."})
    eval_interval: int = field(default=1000, metadata={"help": "Interval (in iterations) at which to evaluate the model on the validation set."})
    eval_batches: int = field(default=20, metadata={"help": "Number of batches to use for evaluation."})
    log_interval: int = field(default=100, metadata={"help": "Interval (in iterations) at which to log training metrics."})

    ckpt_interval: int = field(default=5000, metadata={"help": "Interval (in iterations) at which to save model checkpoints."})
    ckpt_path: str = field(default="checkpoints/checkpoint.pt", metadata={"help": "Directory to save model checkpoints."})
    resume_from_ckpt: Optional[str] = field(default=None, metadata={"help": "Path to a checkpoint to resume training from. If None, training starts from scratch."})

    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})

@dataclass
class WandbConfig:
    """
    Configuration for Weights & Biases (wandb) logging.
    """
    use_wandb: bool = field(default=False, metadata={"help": "Whether to use wandb for logging."})
    project_name: str = field(default="cs336_assignment1", metadata={"help": "Wandb project name."})
    run_name: str = field(default="train", metadata={"help": "Wandb run name. If None, a random name will be generated."})


@dataclass
class TrainConfig:
    """
    Main Train configuration class that aggregates all other configurations.
    """
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

def get_default_config() -> TrainConfig:
    """
    Returns the default training configuration.
    """
    cfg = TrainConfig()
    cfg.model.context_length = cfg.dataset.context_length
    return cfg