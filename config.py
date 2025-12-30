"""
Configuration file for the Tweet Sentiment Extraction project.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    """Data processing configuration."""
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    sample_submission_file: str = "sample_submission.csv"
    max_length: int = 128
    train_split: float = 0.9
    seed: int = 42


@dataclass
class ModelConfig:
    """Model configuration."""
    base_model: str = "google/flan-t5-base"  # Can also use: distilgpt2, facebook/opt-125m
    model_type: str = "seq2seq"  # seq2seq or causal
    max_new_tokens: int = 64

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 4
    do_sample: bool = False


@dataclass
class LoRAConfig:
    """LoRA/PEFT configuration."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "SEQ_2_SEQ_LM"  # or CAUSAL_LM for GPT-style models
    target_modules: Optional[list] = None  # Will be set based on model

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for T5
            self.target_modules = ["q", "v"]


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./models/flan-t5-sentiment-extraction"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = True
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    push_to_hub: bool = False
    report_to: str = "none"  # Can be "wandb" if you want to use W&B


@dataclass
class QuantizationConfig:
    """Quantization configuration for QLoRA."""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    hf_model_id: Optional[str] = None
    streamlit_port: int = 8501
    inference_device: str = "cuda"  # cuda or cpu


class ProjectConfig:
    """Main project configuration."""

    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.lora = LoRAConfig()
        self.training = TrainingConfig()
        self.quantization = QuantizationConfig()
        self.deployment = DeploymentConfig()

        # Load from environment if available
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model config
        if base_model := os.getenv("BASE_MODEL"):
            self.model.base_model = base_model

        # Training config
        if max_length := os.getenv("MAX_LENGTH"):
            self.data.max_length = int(max_length)
        if batch_size := os.getenv("BATCH_SIZE"):
            self.training.per_device_train_batch_size = int(batch_size)
        if lr := os.getenv("LEARNING_RATE"):
            self.training.learning_rate = float(lr)
        if epochs := os.getenv("NUM_EPOCHS"):
            self.training.num_train_epochs = int(epochs)

        # LoRA config
        if lora_r := os.getenv("LORA_R"):
            self.lora.r = int(lora_r)
        if lora_alpha := os.getenv("LORA_ALPHA"):
            self.lora.lora_alpha = int(lora_alpha)
        if lora_dropout := os.getenv("LORA_DROPOUT"):
            self.lora.lora_dropout = float(lora_dropout)

        # Deployment config
        if hf_model_id := os.getenv("HF_MODEL_ID"):
            self.deployment.hf_model_id = hf_model_id
