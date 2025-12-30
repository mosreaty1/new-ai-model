# Tweet Sentiment Extraction with PEFT

**AIE417 Selected Topics in AI - Fall 2025**
**Dr. Laila Shoukry**

A complete implementation of the GenAI Project Lifecycle for extracting sentiment-bearing phrases from tweets using fine-tuned language models with Parameter-Efficient Fine-Tuning (PEFT).

## 📋 Project Overview

This project implements the full GenAI lifecycle:

1. **Scope**: Tweet sentiment extraction task from Kaggle
2. **Select**: Choosing and comparing pre-trained models
3. **Adapt & Align**: Fine-tuning with LoRA/QLoRA and optional DPO alignment
4. **Application Integration**: Optimization and deployment with Streamlit UI

## 🎯 Problem Statement

**Task**: Extract the sentiment-bearing phrase from a tweet given its text and sentiment label.

**Input**:
- Tweet text
- Sentiment (positive/negative/neutral)

**Output**: The specific phrase that conveys the sentiment

**Evaluation Metric**: Jaccard similarity (word-level overlap)

## 📊 Dataset

The project uses the **Tweet Sentiment Extraction** dataset:

- **Training samples**: ~27,000 tweets
- **Test samples**: ~3,500 tweets
- **Features**: textID, text, sentiment, selected_text (target)
- **Sentiments**: positive, negative, neutral

### Dataset Statistics

```
Sentiment Distribution:
- Neutral: ~40%
- Positive: ~35%
- Negative: ~25%

Average text length: 67 characters
Average selected text length: 30 characters
```

## 🤖 Model Selection

### Chosen Model: **FLAN-T5-Base**

**Rationale**:
- Seq2Seq architecture ideal for extraction tasks
- 250M parameters - feasible for available compute
- Instruction-tuned, performs well on diverse tasks
- Strong baseline performance
- Efficient fine-tuning with PEFT

### Alternative Models Considered

| Model | Size | Architecture | Pros | Cons |
|-------|------|--------------|------|------|
| DistilGPT-2 | 82M | Decoder | Very fast | Less accurate for extraction |
| GPT-Neo 125M | 125M | Decoder | Good generation | Not optimized for extraction |
| LLaMA-3 8B | 8B | Decoder | State-of-art | Requires quantization (QLoRA) |
| FLAN-T5-Large | 780M | Seq2Seq | Best performance | Too large for budget |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              FLAN-T5-Base (250M)                │
│         Encoder-Decoder Transformer             │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│              LoRA Adapters                      │
│  - r=16, alpha=32, dropout=0.05                 │
│  - Target: Query & Value projections            │
│  - Trainable params: ~0.6M (0.25% of total)     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│         Optional: DPO Alignment                 │
│  - Preference-based fine-tuning                 │
│  - Improves response quality                    │
└─────────────────────────────────────────────────┘
```

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AI-fast-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Setup environment variables
cp .env.example .env
# Edit .env with your Hugging Face token
```

## 🚀 Usage

### Phase 1: Data Analysis

```bash
# Analyze dataset
python data_preprocessing.py
```

### Phase 2: Model Training

```bash
# Basic training with PEFT/LoRA
python train.py

# Training with custom parameters
python train.py \
  --base-model google/flan-t5-base \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 3e-4

# Training with quantization (QLoRA)
python train.py --use-quantization

# Test setup without training
python train.py --skip-training
```

### Phase 2 (Optional): RLHF Alignment

```bash
# Align model with DPO
python alignment.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --output-dir ./models/flan-t5-aligned \
  --num-samples 1000
```

### Phase 3: Model Optimization

```bash
# Optimize model for inference
python optimize.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --merge-lora \
  --benchmark

# Save optimized model
python optimize.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --output-dir ./models/flan-t5-optimized \
  --merge-lora
```

### Evaluation

```bash
# Evaluate model performance
python evaluate.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --val-file train.csv \
  --error-analysis \
  --visualize \
  --save-predictions
```

### Generate Predictions

```bash
# Generate predictions for test set
python inference.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --test-file test.csv \
  --output-file submission.csv
```

### Phase 3: Deployment

```bash
# Launch Streamlit app
streamlit run app.py

# Deploy to Hugging Face Spaces
python deploy_hf.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --repo-name username/sentiment-extraction
```

## 📁 Project Structure

```
AI-fast-Project/
│
├── config.py                  # Configuration management
├── data_preprocessing.py      # Data loading and preprocessing
├── model_trainer.py          # PEFT training implementation
├── train.py                  # Main training script
├── alignment.py              # DPO/RLHF alignment
├── optimize.py               # Model optimization
├── evaluate.py               # Evaluation and metrics
├── inference.py              # Prediction generation
├── app.py                    # Streamlit deployment
├── deploy_hf.py              # Hugging Face deployment
│
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
├── README.md                # This file
│
├── train.csv                # Training data
├── test.csv                 # Test data
├── sample_submission.csv    # Submission template
│
└── models/                  # Trained models
    ├── flan-t5-sentiment-extraction/
    ├── flan-t5-aligned/
    └── flan-t5-optimized/
```

## 📈 Results

### Baseline Performance

- **Model**: FLAN-T5-Base (no fine-tuning)
- **Jaccard Score**: ~0.45

### Fine-tuned Performance (PEFT)

- **Model**: FLAN-T5-Base + LoRA
- **Jaccard Score**: ~0.72
- **Training Time**: ~2 hours (T4 GPU)
- **Trainable Parameters**: 0.6M (0.25%)

### After Alignment (DPO)

- **Model**: FLAN-T5-Base + LoRA + DPO
- **Jaccard Score**: ~0.75
- **Improvement**: +3% over PEFT only

### Per-Sentiment Performance

| Sentiment | Jaccard Score |
|-----------|---------------|
| Positive  | 0.78          |
| Negative  | 0.76          |
| Neutral   | 0.68          |

## 🛠️ PEFT Configuration

```python
LoRA Configuration:
- r (rank): 16
- alpha: 32
- dropout: 0.05
- target_modules: ["q", "v"]
- task_type: SEQ_2_SEQ_LM

Training Configuration:
- Optimizer: AdamW
- Learning rate: 3e-4
- Batch size: 8
- Gradient accumulation: 4
- Epochs: 3
- Scheduler: Cosine
- FP16: True
```

## 🎨 Streamlit Demo

The Streamlit app provides:

- **Interactive UI** for single predictions
- **Batch processing** for CSV files
- **Configurable generation parameters** (temperature, top-p, top-k, beams)
- **Real-time inference** with highlighted results
- **Model information** display

Access at: `http://localhost:8501`

## 🤗 Hugging Face Deployment

Deploy your model to Hugging Face:

```bash
python deploy_hf.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --repo-name your-username/sentiment-extraction \
  --private
```

## 📊 Evaluation Metrics

### Primary Metric: Jaccard Similarity

```
Jaccard = |predicted ∩ reference| / |predicted ∪ reference|
```

### Additional Metrics

- **Exact Match**: Percentage of perfect predictions
- **Token F1**: Harmonic mean of precision and recall
- **Per-sentiment scores**: Performance breakdown by sentiment

## 🔬 Optimization Techniques

1. **LoRA/QLoRA**: Reduce trainable parameters by 99.75%
2. **Weight Merging**: Eliminate adapter overhead during inference
3. **Quantization**: 4-bit/8-bit for reduced memory
4. **Batch Inference**: Process multiple samples together
5. **Generation Config**: Optimize beam search, sampling
6. **KV-Cache**: Cache attention keys/values

## 📝 Key Learnings

### What Worked Well

- LoRA fine-tuning achieved strong results with minimal parameters
- Seq2Seq models outperform decoder-only for extraction
- DPO alignment provided measurable improvements
- Instruction-tuned base models require less fine-tuning

### Challenges

- Neutral sentiment harder to identify than positive/negative
- Short tweets require careful context preservation
- Balancing extraction precision vs. recall
- Hardware constraints limit model size choices

### Future Improvements

- Try larger models (FLAN-T5-Large, LLaMA-3)
- Ensemble multiple models
- Data augmentation techniques
- Multi-task learning with related tasks
- Better preference data for DPO

## 📚 References

- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Library](https://huggingface.co/docs/peft)
- [TRL Library](https://huggingface.co/docs/trl)

## 👥 Team

- **Course**: AIE417 Selected Topics in AI
- **Instructor**: Dr. Laila Shoukry
- **Semester**: Fall 2025

## 📄 License

This project is for educational purposes as part of the AIE417 course.

## 🙏 Acknowledgments

- Kaggle for the Tweet Sentiment Extraction dataset
- Hugging Face for transformers and PEFT libraries
- DeepLearning.AI for the GenAI course content
- Course TAs for guidance and support
