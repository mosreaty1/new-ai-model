"""
Main training script for Tweet Sentiment Extraction project.
Phase 2: Model Fine-tuning with PEFT
"""
import argparse
import os
from config import ProjectConfig
from data_preprocessing import SentimentDataPreprocessor
from model_trainer import SentimentExtractionTrainer


def main(args):
    """Main training function."""
    print("=" * 80)
    print("AIE417 - Tweet Sentiment Extraction Project")
    print("Phase 2: Model Fine-tuning with PEFT/LoRA")
    print("=" * 80)

    # Initialize configuration
    config = ProjectConfig()

    # Override config with command line arguments
    if args.base_model:
        config.model.base_model = args.base_model
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    # Step 1: Load and preprocess data
    print("\n" + "=" * 80)
    print("STEP 1: Data Loading and Preprocessing")
    print("=" * 80)

    preprocessor = SentimentDataPreprocessor(max_length=config.data.max_length)
    train_df, test_df = preprocessor.load_data(
        config.data.train_file,
        config.data.test_file
    )

    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    # Analyze dataset
    print("\nDataset Statistics:")
    stats = preprocessor.analyze_dataset(train_df)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Prepare dataset
    dataset = preprocessor.prepare_dataset(
        train_df,
        is_train=True,
        train_split=config.data.train_split,
        seed=config.data.seed
    )

    print(f"\nTrain split: {len(dataset['train'])} samples")
    print(f"Validation split: {len(dataset['validation'])} samples")

    print("\nSample data:")
    print(f"  Input: {dataset['train'][0]['input_text']}")
    print(f"  Target: {dataset['train'][0]['target_text']}")

    # Step 2: Load pre-trained model
    print("\n" + "=" * 80)
    print("STEP 2: Loading Pre-trained Model")
    print("=" * 80)

    trainer = SentimentExtractionTrainer(config)
    trainer.load_model_and_tokenizer(use_quantization=args.use_quantization)

    # Step 3: Setup PEFT/LoRA
    print("\n" + "=" * 80)
    print("STEP 3: Setting up PEFT/LoRA")
    print("=" * 80)

    print(f"LoRA Configuration:")
    print(f"  r: {config.lora.r}")
    print(f"  alpha: {config.lora.lora_alpha}")
    print(f"  dropout: {config.lora.lora_dropout}")
    print(f"  target_modules: {config.lora.target_modules}")

    trainer.setup_peft()

    # Step 4: Fine-tune the model
    print("\n" + "=" * 80)
    print("STEP 4: Fine-tuning Model with PEFT")
    print("=" * 80)

    if not args.skip_training:
        trainer.train(dataset)

        # Save the model
        print("\n" + "=" * 80)
        print("STEP 5: Saving Fine-tuned Model")
        print("=" * 80)

        trainer.save_model()
        print(f"Model saved to: {config.training.output_dir}")
    else:
        print("Skipping training (--skip-training flag set)")

    # Step 5: Evaluate on validation set
    print("\n" + "=" * 80)
    print("STEP 6: Evaluation on Validation Set")
    print("=" * 80)

    if not args.skip_training and trainer.trainer:
        metrics = trainer.trainer.evaluate()
        print("\nValidation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment extraction model")

    # Model arguments
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model to use (default: google/flan-t5-base)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for trained model"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (for testing setup)"
    )

    args = parser.parse_args()
    main(args)
