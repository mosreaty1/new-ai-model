"""
RLHF/DPO Alignment module (Optional Phase 2 component).
Implements Direct Preference Optimization (DPO) for model alignment.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
import pandas as pd
from typing import List, Dict
import random


class PreferenceDataGenerator:
    """Generate preference pairs for DPO training."""

    def __init__(self, train_df: pd.DataFrame):
        self.train_df = train_df

    def create_preference_pairs(self, num_samples: int = 1000) -> Dataset:
        """
        Create preference pairs for DPO.
        For each sample, create a chosen (ground truth) and rejected (corrupted) response.
        """
        preference_data = []

        # Sample from training data
        samples = self.train_df.sample(min(num_samples, len(self.train_df)), random_state=42)

        for _, row in samples.iterrows():
            text = row['text']
            sentiment = row['sentiment']
            selected_text = row['selected_text']

            # Create prompt
            prompt = f"Extract the {sentiment} sentiment phrase from: {text}"

            # Chosen: ground truth
            chosen = selected_text

            # Rejected: create a corrupted version
            # Strategy: use a random different phrase from the text, or truncate/extend
            rejected = self._create_rejected_response(text, selected_text, sentiment)

            preference_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })

        return Dataset.from_list(preference_data)

    def _create_rejected_response(self, text: str, selected_text: str, sentiment: str) -> str:
        """Create a rejected (inferior) response."""
        strategies = [
            lambda: text,  # Return full text (too long)
            lambda: selected_text[:len(selected_text)//2],  # Truncate
            lambda: self._get_random_phrase(text, selected_text),  # Random phrase
            lambda: "",  # Empty response
        ]

        # Choose a random corruption strategy
        rejected = random.choice(strategies)()

        # Make sure rejected is different from chosen
        if rejected == selected_text:
            rejected = text

        return rejected

    def _get_random_phrase(self, text: str, avoid: str) -> str:
        """Get a random phrase from text, avoiding the correct one."""
        words = text.split()
        if len(words) < 3:
            return text

        # Take a random slice
        start = random.randint(0, max(0, len(words) - 3))
        end = random.randint(start + 1, min(start + 5, len(words)))

        phrase = ' '.join(words[start:end])

        # If it matches the correct answer, return full text
        if phrase == avoid:
            return text

        return phrase


def train_with_dpo(
    model_path: str,
    base_model_name: str,
    train_df: pd.DataFrame,
    output_dir: str,
    num_samples: int = 1000,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-5
):
    """Train model with Direct Preference Optimization."""

    print("=" * 80)
    print("DPO Alignment Training")
    print("=" * 80)

    # Generate preference dataset
    print("\nGenerating preference pairs...")
    pref_generator = PreferenceDataGenerator(train_df)
    pref_dataset = pref_generator.create_preference_pairs(num_samples=num_samples)

    print(f"Generated {len(pref_dataset)} preference pairs")
    print(f"\nSample preference pair:")
    print(f"  Prompt: {pref_dataset[0]['prompt']}")
    print(f"  Chosen: {pref_dataset[0]['chosen']}")
    print(f"  Rejected: {pref_dataset[0]['rejected']}")

    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, model_path)

    # Merge and unload for DPO (DPO works better with merged weights)
    print("Merging LoRA weights for DPO training...")
    model = model.merge_and_unload()

    # Create reference model (copy of the current model)
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    ref_model.load_state_dict(model.state_dict())

    # DPO configuration
    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=4,
        logging_steps=10,
        save_steps=100,
        warmup_steps=50,
        bf16=True,
        remove_unused_columns=False,
    )

    # Initialize DPO trainer
    print("\nInitializing DPO trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=pref_dataset,
        tokenizer=tokenizer,
        beta=0.1,  # KL divergence coefficient
    )

    # Train
    print("\nStarting DPO training...")
    dpo_trainer.train()

    # Save aligned model
    print(f"\nSaving aligned model to {output_dir}...")
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 80)
    print("DPO alignment completed!")
    print("=" * 80)


def main(args):
    """Main alignment function."""

    # Load training data
    print(f"Loading training data from {args.train_file}...")
    train_df = pd.read_csv(args.train_file)

    # Handle missing values
    train_df['selected_text'] = train_df['selected_text'].fillna(train_df['text'])

    print(f"Loaded {len(train_df)} training samples")

    # Run DPO training
    train_with_dpo(
        model_path=args.model_path,
        base_model_name=args.base_model,
        train_df=train_df,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align model with DPO")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model to align"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/flan-t5-base",
        help="Base model name"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="train.csv",
        help="Training data file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/flan-t5-aligned",
        help="Output directory for aligned model"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of preference pairs to generate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )

    args = parser.parse_args()
    main(args)
