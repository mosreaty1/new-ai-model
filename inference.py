"""
Inference script for generating predictions on test set.
"""
import argparse
import pandas as pd
from tqdm import tqdm
from config import ProjectConfig
from model_trainer import SentimentExtractionTrainer


def main(args):
    """Generate predictions on test set."""
    print("=" * 80)
    print("AIE417 - Tweet Sentiment Extraction")
    print("Generating Predictions")
    print("=" * 80)

    # Load configuration
    config = ProjectConfig()

    # Load test data
    print(f"\nLoading test data from {args.test_file}...")
    test_df = pd.read_csv(args.test_file)
    print(f"Test samples: {len(test_df)}")

    # Load fine-tuned model
    print(f"\nLoading model from {args.model_path}...")
    trainer = SentimentExtractionTrainer(config)
    trainer.load_finetuned_model(args.model_path)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []

    batch_size = args.batch_size
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch = test_df.iloc[i:i + batch_size]
        batch_texts = batch['text'].tolist()
        batch_sentiments = batch['sentiment'].tolist()

        batch_predictions = trainer.generate_predictions(batch_texts, batch_sentiments)
        predictions.extend(batch_predictions)

    # Create submission file
    submission_df = pd.DataFrame({
        'textID': test_df['textID'],
        'selected_text': predictions
    })

    # Save submission
    print(f"\nSaving predictions to {args.output_file}...")
    submission_df.to_csv(args.output_file, index=False)

    print("\n" + "=" * 80)
    print(f"Predictions saved successfully!")
    print(f"Total predictions: {len(predictions)}")
    print(f"Output file: {args.output_file}")
    print("=" * 80)

    # Show sample predictions
    print("\nSample Predictions:")
    for i in range(min(5, len(test_df))):
        print(f"\n{i+1}. Text: {test_df.iloc[i]['text']}")
        print(f"   Sentiment: {test_df.iloc[i]['sentiment']}")
        print(f"   Prediction: {predictions[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions on test set")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="test.csv",
        help="Path to test file"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="submission.csv",
        help="Path to output submission file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )

    args = parser.parse_args()
    main(args)
