"""
Evaluation script for model performance analysis.
"""
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import SentimentDataPreprocessor
from config import ProjectConfig
from model_trainer import SentimentExtractionTrainer


def calculate_metrics(predictions: List[str], references: List[str], sentiments: List[str]) -> Dict:
    """Calculate comprehensive evaluation metrics."""

    # Jaccard similarity
    jaccard_scores = []
    exact_matches = 0
    f1_scores = []

    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())

        # Jaccard
        if len(pred_words.union(ref_words)) == 0:
            jaccard = 1.0 if len(pred_words) == 0 and len(ref_words) == 0 else 0.0
        else:
            jaccard = len(pred_words.intersection(ref_words)) / len(pred_words.union(ref_words))
        jaccard_scores.append(jaccard)

        # Exact match
        if pred.lower().strip() == ref.lower().strip():
            exact_matches += 1

        # Token F1
        if len(pred_words) + len(ref_words) == 0:
            f1 = 1.0
        else:
            precision = len(pred_words.intersection(ref_words)) / len(pred_words) if len(pred_words) > 0 else 0
            recall = len(pred_words.intersection(ref_words)) / len(ref_words) if len(ref_words) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    # Overall metrics
    metrics = {
        'jaccard_mean': np.mean(jaccard_scores),
        'jaccard_std': np.std(jaccard_scores),
        'exact_match_ratio': exact_matches / len(predictions),
        'f1_mean': np.mean(f1_scores),
        'f1_std': np.std(f1_scores)
    }

    # Per-sentiment metrics
    for sentiment in ['positive', 'negative', 'neutral']:
        sentiment_mask = [s == sentiment for s in sentiments]
        if sum(sentiment_mask) > 0:
            sentiment_jaccards = [j for j, m in zip(jaccard_scores, sentiment_mask) if m]
            metrics[f'jaccard_{sentiment}'] = np.mean(sentiment_jaccards)

    return metrics, jaccard_scores


def analyze_errors(
    texts: List[str],
    predictions: List[str],
    references: List[str],
    sentiments: List[str],
    jaccard_scores: List[float],
    top_k: int = 10
):
    """Analyze and display worst predictions."""

    # Get worst predictions
    sorted_indices = np.argsort(jaccard_scores)

    print("\n" + "=" * 80)
    print(f"Top {top_k} Worst Predictions:")
    print("=" * 80)

    for i, idx in enumerate(sorted_indices[:top_k]):
        print(f"\n{i+1}. Jaccard Score: {jaccard_scores[idx]:.4f}")
        print(f"   Sentiment: {sentiments[idx]}")
        print(f"   Text: {texts[idx][:100]}...")
        print(f"   Reference: {references[idx]}")
        print(f"   Prediction: {predictions[idx]}")
        print("-" * 80)


def plot_results(jaccard_scores: List[float], sentiments: List[str], output_dir: str):
    """Create visualizations of results."""

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Jaccard score distribution
    axes[0, 0].hist(jaccard_scores, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Jaccard Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Jaccard Scores')
    axes[0, 0].axvline(np.mean(jaccard_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(jaccard_scores):.4f}')
    axes[0, 0].legend()

    # 2. Box plot by sentiment
    sentiment_scores = {s: [] for s in ['positive', 'negative', 'neutral']}
    for score, sent in zip(jaccard_scores, sentiments):
        sentiment_scores[sent].append(score)

    axes[0, 1].boxplot([sentiment_scores['positive'], sentiment_scores['negative'],
                        sentiment_scores['neutral']],
                       labels=['Positive', 'Negative', 'Neutral'])
    axes[0, 1].set_ylabel('Jaccard Score')
    axes[0, 1].set_title('Jaccard Scores by Sentiment')

    # 3. Violin plot by sentiment
    data_for_violin = []
    labels_for_violin = []
    for sent in ['positive', 'negative', 'neutral']:
        data_for_violin.extend(sentiment_scores[sent])
        labels_for_violin.extend([sent.capitalize()] * len(sentiment_scores[sent]))

    df_violin = pd.DataFrame({'Score': data_for_violin, 'Sentiment': labels_for_violin})
    sns.violinplot(data=df_violin, x='Sentiment', y='Score', ax=axes[1, 0])
    axes[1, 0].set_title('Score Distribution by Sentiment (Violin Plot)')

    # 4. Score ranges
    score_ranges = {
        '0.0-0.2': sum(1 for s in jaccard_scores if s < 0.2),
        '0.2-0.4': sum(1 for s in jaccard_scores if 0.2 <= s < 0.4),
        '0.4-0.6': sum(1 for s in jaccard_scores if 0.4 <= s < 0.6),
        '0.6-0.8': sum(1 for s in jaccard_scores if 0.6 <= s < 0.8),
        '0.8-1.0': sum(1 for s in jaccard_scores if s >= 0.8),
    }

    axes[1, 1].bar(score_ranges.keys(), score_ranges.values(), edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Score Range')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Score Ranges')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/evaluation_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir}/evaluation_results.png")


def main(args):
    """Main evaluation function."""
    print("=" * 80)
    print("Model Evaluation")
    print("=" * 80)

    # Load validation data
    print(f"\nLoading validation data from {args.val_file}...")
    val_df = pd.read_csv(args.val_file)
    val_df['selected_text'] = val_df['selected_text'].fillna(val_df['text'])

    print(f"Validation samples: {len(val_df)}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    config = ProjectConfig()
    trainer = SentimentExtractionTrainer(config)
    trainer.load_finetuned_model(args.model_path)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = trainer.generate_predictions(
        val_df['text'].tolist(),
        val_df['sentiment'].tolist()
    )

    # Calculate metrics
    print("\n" + "=" * 80)
    print("Evaluation Metrics")
    print("=" * 80)

    metrics, jaccard_scores = calculate_metrics(
        predictions,
        val_df['selected_text'].tolist(),
        val_df['sentiment'].tolist()
    )

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Error analysis
    if args.error_analysis:
        analyze_errors(
            val_df['text'].tolist(),
            predictions,
            val_df['selected_text'].tolist(),
            val_df['sentiment'].tolist(),
            jaccard_scores,
            top_k=args.top_k_errors
        )

    # Create visualizations
    if args.visualize:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        plot_results(jaccard_scores, val_df['sentiment'].tolist(), args.output_dir)

    # Save detailed results
    if args.save_predictions:
        results_df = val_df.copy()
        results_df['prediction'] = predictions
        results_df['jaccard_score'] = jaccard_scores

        output_file = f"{args.output_dir}/evaluation_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="train.csv",
        help="Validation data file (will use last 10%)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--error-analysis",
        action="store_true",
        help="Perform error analysis"
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=10,
        help="Number of worst predictions to show"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save detailed predictions to CSV"
    )

    args = parser.parse_args()
    main(args)
