"""
Model training module with PEFT/LoRA fine-tuning.
"""
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from datasets import DatasetDict
import numpy as np
from typing import Dict, Optional
from config import ProjectConfig
from data_preprocessing import SentimentDataPreprocessor


class SentimentExtractionTrainer:
    """Trainer for sentiment extraction with PEFT."""

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.trainer = None

    def load_model_and_tokenizer(self, use_quantization: bool = False):
        """Load pre-trained model and tokenizer."""
        print(f"Loading model: {self.config.model.base_model}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model,
            use_fast=True
        )

        # Configure quantization if needed
        quantization_config = None
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.quantization.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
            )

        # Load model based on type
        if self.config.model.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model.base_model,
                quantization_config=quantization_config,
                device_map="auto" if use_quantization else None,
                torch_dtype=torch.float16 if use_quantization else torch.float32,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.base_model,
                quantization_config=quantization_config,
                device_map="auto" if use_quantization else None,
                torch_dtype=torch.float16 if use_quantization else torch.float32,
            )

        # Add pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        print(f"Model loaded. Parameters: {self.model.num_parameters():,}")

    def setup_peft(self):
        """Configure and apply PEFT/LoRA."""
        print("Setting up PEFT/LoRA...")

        # Prepare model for k-bit training if quantized
        if hasattr(self.model, 'is_loaded_in_4bit') and self.model.is_loaded_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        task_type = TaskType.SEQ_2_SEQ_LM if self.config.model.model_type == "seq2seq" else TaskType.CAUSAL_LM

        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=task_type,
            target_modules=self.config.lora.target_modules,
        )

        # Apply PEFT
        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

    def preprocess_function(self, examples: Dict):
        """Tokenize inputs and targets."""
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=self.config.data.max_length,
            truncation=True,
            padding='max_length',
        )

        # Tokenize targets
        if 'target_text' in examples:
            labels = self.tokenizer(
                examples['target_text'],
                max_length=self.config.model.max_new_tokens,
                truncation=True,
                padding='max_length',
            )
            model_inputs['labels'] = labels['input_ids']

        return model_inputs

    def train(self, dataset: DatasetDict):
        """Train the model with PEFT."""
        print("Preparing dataset...")

        # Tokenize datasets
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset['train'].column_names,
        )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.peft_model,
            padding=True,
        )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            fp16=self.config.training.fp16,
            optim=self.config.training.optim,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            report_to=self.config.training.report_to,
            evaluation_strategy="steps",
            predict_with_generate=True,
            generation_max_length=self.config.model.max_new_tokens,
        )

        # Initialize trainer
        self.trainer = Seq2SeqTrainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        print("Starting training...")
        self.trainer.train()

        print("Training completed!")

    def compute_metrics(self, eval_preds):
        """Compute metrics during evaluation."""
        predictions, labels = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Compute Jaccard similarity
        preprocessor = SentimentDataPreprocessor()
        metrics = preprocessor.compute_metrics(decoded_preds, decoded_labels)

        return metrics

    def save_model(self, output_dir: Optional[str] = None):
        """Save the fine-tuned model."""
        if output_dir is None:
            output_dir = self.config.training.output_dir

        print(f"Saving model to {output_dir}")
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_finetuned_model(self, model_path: str):
        """Load a fine-tuned model."""
        from peft import PeftModel

        print(f"Loading fine-tuned model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.config.model.model_type == "seq2seq":
            base_model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model.base_model)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(self.config.model.base_model)

        self.peft_model = PeftModel.from_pretrained(base_model, model_path)

    def generate_predictions(self, texts: list, sentiments: list) -> list:
        """Generate predictions for a list of texts."""
        self.peft_model.eval()

        preprocessor = SentimentDataPreprocessor()
        prompts = [
            preprocessor.create_prompt(text, sentiment)
            for text, sentiment in zip(texts, sentiments)
        ]

        predictions = []
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=self.config.data.max_length,
                    truncation=True,
                ).to(self.peft_model.device)

                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=self.config.model.max_new_tokens,
                    num_beams=self.config.model.num_beams,
                    temperature=self.config.model.temperature,
                    top_p=self.config.model.top_p,
                    top_k=self.config.model.top_k,
                    do_sample=self.config.model.do_sample,
                )

                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predictions.append(prediction.strip())

        return predictions


if __name__ == "__main__":
    # Test the trainer setup
    config = ProjectConfig()
    trainer = SentimentExtractionTrainer(config)

    print("Testing model loading...")
    trainer.load_model_and_tokenizer(use_quantization=False)

    print("\nTesting PEFT setup...")
    trainer.setup_peft()

    print("\nTrainer setup completed successfully!")
