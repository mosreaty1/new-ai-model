"""
Data preprocessing module for Tweet Sentiment Extraction.
"""
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from typing import Dict, Tuple
import re


class SentimentDataPreprocessor:
    """Preprocesses tweet sentiment extraction data."""

    def __init__(self, max_length: int = 128):
        self.max_length = max_length

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""

        # Convert to string and strip
        text = str(text).strip()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text

    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test datasets."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Clean the data
        train_df['text'] = train_df['text'].apply(self.clean_text)
        train_df['selected_text'] = train_df['selected_text'].apply(self.clean_text)
        test_df['text'] = test_df['text'].apply(self.clean_text)

        # Handle missing values
        train_df['selected_text'] = train_df['selected_text'].fillna(train_df['text'])

        return train_df, test_df

    def create_prompt(self, text: str, sentiment: str) -> str:
        """
        Create a prompt for the model.
        Format: "Extract the sentiment phrase: [SENTIMENT] Text: [TEXT]"
        """
        prompt = f"Extract the {sentiment} sentiment phrase from: {text}"
        return prompt

    def prepare_dataset(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        train_split: float = 0.9,
        seed: int = 42
    ) -> DatasetDict:
        """Prepare dataset for training."""

        # Create prompts
        df['input_text'] = df.apply(
            lambda row: self.create_prompt(row['text'], row['sentiment']),
            axis=1
        )

        if is_train:
            df['target_text'] = df['selected_text']

            # Split into train and validation
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
            split_idx = int(len(df) * train_split)

            train_data = df[:split_idx]
            val_data = df[split_idx:]

            dataset = DatasetDict({
                'train': Dataset.from_pandas(train_data[['input_text', 'target_text']]),
                'validation': Dataset.from_pandas(val_data[['input_text', 'target_text']])
            })
        else:
            dataset = Dataset.from_pandas(df[['textID', 'input_text']])

        return dataset

    def compute_metrics(self, predictions: list, references: list) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        Uses Jaccard similarity (word-level overlap) as the main metric.
        """
        jaccard_scores = []

        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())

            if len(pred_words.union(ref_words)) == 0:
                jaccard = 1.0 if len(pred_words) == 0 and len(ref_words) == 0 else 0.0
            else:
                jaccard = len(pred_words.intersection(ref_words)) / len(pred_words.union(ref_words))

            jaccard_scores.append(jaccard)

        return {
            'jaccard': np.mean(jaccard_scores),
            'std': np.std(jaccard_scores)
        }

    def analyze_dataset(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset statistics."""
        stats = {
            'total_samples': len(df),
            'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
            'avg_text_length': df['text'].str.len().mean(),
            'avg_selected_length': df['selected_text'].str.len().mean() if 'selected_text' in df.columns else None,
            'text_length_stats': df['text'].str.len().describe().to_dict(),
        }

        if 'selected_text' in df.columns:
            stats['selected_length_stats'] = df['selected_text'].str.len().describe().to_dict()

        return stats


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = SentimentDataPreprocessor()

    train_df, test_df = preprocessor.load_data("train.csv", "test.csv")

    print("Train Dataset Stats:")
    train_stats = preprocessor.analyze_dataset(train_df)
    for key, value in train_stats.items():
        print(f"{key}: {value}")

    print("\nTest Dataset Stats:")
    test_stats = preprocessor.analyze_dataset(test_df)
    for key, value in test_stats.items():
        print(f"{key}: {value}")

    # Create dataset
    dataset = preprocessor.prepare_dataset(train_df, is_train=True)
    print(f"\nDataset splits: {dataset}")
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['validation'])}")
    print(f"\nSample input: {dataset['train'][0]['input_text']}")
    print(f"Sample target: {dataset['train'][0]['target_text']}")
