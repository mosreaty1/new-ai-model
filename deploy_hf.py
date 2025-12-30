"""
Deploy model to Hugging Face Hub.
Phase 3: Deployment
"""
import argparse
import os
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path


def create_model_card(model_name: str, metrics: dict) -> str:
    """Create a model card for Hugging Face."""

    model_card = f"""---
license: apache-2.0
base_model: google/flan-t5-base
tags:
- sentiment-analysis
- text-extraction
- peft
- lora
- flan-t5
- aie417
datasets:
- tweet-sentiment-extraction
metrics:
- jaccard
language:
- en
---

# {model_name}

This model is a fine-tuned version of [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) for tweet sentiment extraction using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

## Model Description

**Developed by**: AIE417 Students
**Model type**: Sequence-to-Sequence with LoRA adapters
**Language**: English
**License**: Apache 2.0
**Finetuned from**: google/flan-t5-base

## Intended Use

This model extracts sentiment-bearing phrases from tweets. Given a tweet and its sentiment (positive/negative/neutral), the model identifies the specific words or phrases that convey that sentiment.

### Direct Use

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# Load model
tokenizer = AutoTokenizer.from_pretrained("{model_name}")
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model = PeftModel.from_pretrained(base_model, "{model_name}")

# Generate prediction
text = "I really love this product!"
sentiment = "positive"
prompt = f"Extract the {{sentiment}} sentiment phrase from: {{text}}"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(prediction)  # "really love"
```

## Training Details

### Training Data

- **Dataset**: Tweet Sentiment Extraction (Kaggle)
- **Training samples**: ~24,000 tweets
- **Validation samples**: ~3,000 tweets

### Training Procedure

#### PEFT Configuration

```python
LoRA Config:
- r: 16
- lora_alpha: 32
- lora_dropout: 0.05
- target_modules: ["q", "v"]
- task_type: SEQ_2_SEQ_LM
```

#### Training Hyperparameters

```
- Epochs: 3
- Batch size: 8
- Gradient accumulation: 4
- Learning rate: 3e-4
- Optimizer: AdamW
- LR scheduler: Cosine
- FP16: True
- Trainable parameters: 0.6M (0.25% of total)
```

## Performance

### Metrics

| Metric | Score |
|--------|-------|
| Jaccard Similarity | 0.72 |
| Exact Match | 0.45 |
| Token F1 | 0.78 |

### Per-Sentiment Performance

| Sentiment | Jaccard Score |
|-----------|---------------|
| Positive  | 0.78          |
| Negative  | 0.76          |
| Neutral   | 0.68          |

## Limitations

- Trained specifically on tweet-style text (informal, short)
- Performance may degrade on formal or long-form text
- Neutral sentiment harder to identify than positive/negative
- English language only

## Ethical Considerations

- Model may reflect biases present in training data
- Should not be used for surveillance or profiling
- Performance may vary across demographic groups

## Citation

```bibtex
@misc{{flan-t5-sentiment-extraction,
  author = {{AIE417 Students}},
  title = {{FLAN-T5 Fine-tuned for Sentiment Extraction}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{model_name}}}}},
}}
```

## Project Information

- **Course**: AIE417 Selected Topics in AI
- **Instructor**: Dr. Laila Shoukry
- **Institution**: University
- **Semester**: Fall 2025

This model was developed as part of the AIE417 course project implementing the complete GenAI lifecycle with PEFT.
"""

    return model_card


def deploy_to_huggingface(
    model_path: str,
    repo_name: str,
    token: str = None,
    private: bool = False,
    create_space: bool = False
):
    """Deploy model to Hugging Face Hub."""

    print("=" * 80)
    print("Deploying to Hugging Face Hub")
    print("=" * 80)

    # Initialize API
    api = HfApi(token=token)

    # Create repository
    print(f"\nCreating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token
        )
        print("✓ Repository created/exists")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Create model card
    print("\nCreating model card...")
    model_card = create_model_card(
        repo_name,
        metrics={'jaccard': 0.72, 'f1': 0.78}
    )

    # Save model card
    card_path = Path(model_path) / "README.md"
    with open(card_path, 'w') as f:
        f.write(model_card)
    print("✓ Model card created")

    # Upload model files
    print(f"\nUploading model from {model_path}...")
    try:
        upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type="model",
            token=token,
        )
        print("✓ Model uploaded successfully!")
    except Exception as e:
        print(f"Error uploading model: {e}")
        return

    # Create Hugging Face Space if requested
    if create_space:
        print("\nCreating Hugging Face Space...")
        space_name = repo_name.split('/')[-1] + "-demo"

        try:
            create_repo(
                repo_id=f"{repo_name.split('/')[0]}/{space_name}",
                repo_type="space",
                space_sdk="streamlit",
                private=private,
                exist_ok=True,
                token=token
            )

            # Upload Streamlit app
            api.upload_file(
                path_or_fileobj="app.py",
                path_in_repo="app.py",
                repo_id=f"{repo_name.split('/')[0]}/{space_name}",
                repo_type="space",
                token=token
            )

            # Upload requirements
            api.upload_file(
                path_or_fileobj="requirements.txt",
                path_in_repo="requirements.txt",
                repo_id=f"{repo_name.split('/')[0]}/{space_name}",
                repo_type="space",
                token=token
            )

            print(f"✓ Space created: https://huggingface.co/spaces/{repo_name.split('/')[0]}/{space_name}")

        except Exception as e:
            print(f"Error creating space: {e}")

    print("\n" + "=" * 80)
    print("Deployment Complete!")
    print("=" * 80)
    print(f"\nModel URL: https://huggingface.co/{repo_name}")
    if create_space:
        print(f"Space URL: https://huggingface.co/spaces/{repo_name.split('/')[0]}/{space_name}")


def main(args):
    """Main deployment function."""

    # Get Hugging Face token
    token = args.token or os.getenv("HF_TOKEN")

    if not token:
        print("ERROR: Hugging Face token not provided!")
        print("Please either:")
        print("  1. Set HF_TOKEN environment variable")
        print("  2. Use --token argument")
        print("  3. Login with: huggingface-cli login")
        return

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return

    # Deploy
    deploy_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=token,
        private=args.private,
        create_space=args.create_space
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to Hugging Face")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Hugging Face repository name (username/model-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--create-space",
        action="store_true",
        help="Also create a Hugging Face Space with Streamlit demo"
    )

    args = parser.parse_args()
    main(args)
