# Quick Start Guide

Get your Tweet Sentiment Extraction model running in minutes!

## 🚀 Option 1: Quick Demo (No Training)

If you just want to see the app without training:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download pre-trained model (if available)
# huggingface-cli download username/flan-t5-sentiment-extraction --local-dir ./models/flan-t5-sentiment-extraction

# 3. Run demo app
streamlit run app.py
```

## 🔧 Option 2: Full Training Pipeline

### Step 1: Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Explore Data

```bash
# Analyze dataset statistics
python data_preprocessing.py
```

**Output**: Dataset stats, sentiment distribution, text lengths

### Step 3: Train Model

```bash
# Quick training (small test run)
python train.py --epochs 1 --batch-size 4

# Full training (recommended)
python train.py --epochs 3 --batch-size 8

# Training with quantization (for limited GPU memory)
python train.py --use-quantization
```

**Expected time**: 2-3 hours on T4 GPU

### Step 4: Evaluate

```bash
python evaluate.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --error-analysis \
  --visualize
```

**Output**: Metrics, error analysis, visualization plots

### Step 5: Generate Predictions

```bash
python inference.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --test-file test.csv \
  --output-file submission.csv
```

**Output**: `submission.csv` ready for Kaggle

### Step 6: Deploy

```bash
# Local deployment
streamlit run app.py

# Hugging Face deployment
python deploy_hf.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --repo-name your-username/sentiment-model
```

## 🎯 Optional: Advanced Features

### RLHF Alignment (DPO)

```bash
python alignment.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --output-dir ./models/flan-t5-aligned \
  --num-samples 1000
```

### Model Optimization

```bash
python optimize.py \
  --model-path ./models/flan-t5-sentiment-extraction \
  --merge-lora \
  --benchmark \
  --output-dir ./models/flan-t5-optimized
```

## 📊 Expected Results

| Phase | Metric | Expected Value |
|-------|--------|----------------|
| Baseline | Jaccard | ~0.45 |
| After PEFT | Jaccard | ~0.72 |
| After DPO | Jaccard | ~0.75 |
| Kaggle Score | Jaccard | ~0.71 |

## ⚠️ Common Issues

### Issue: CUDA Out of Memory

**Solutions**:
1. Reduce batch size: `--batch-size 4`
2. Use quantization: `--use-quantization`
3. Increase gradient accumulation: Edit `config.py`

### Issue: Slow Training

**Solutions**:
1. Use FP16: Already enabled by default
2. Reduce epochs: `--epochs 1` for testing
3. Use smaller model: Change to `flan-t5-small` in config

### Issue: Model Not Found

**Solution**:
```bash
# Ensure model is downloaded
python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/flan-t5-base')"
```

## 🎓 Learning Path

### Beginner Track
1. Run `data_preprocessing.py` - understand the data
2. Run `train.py --skip-training` - test setup
3. Run `train.py --epochs 1` - short training
4. Run `app.py` - see the demo

### Intermediate Track
1. Full training (3 epochs)
2. Evaluation and error analysis
3. Generate Kaggle submission
4. Deploy to Hugging Face

### Advanced Track
1. Experiment with hyperparameters
2. Implement DPO alignment
3. Optimize for inference
4. Create custom modifications

## 📚 Next Steps

- Read `PROJECT_REPORT.md` for detailed analysis
- Check `README.md` for comprehensive documentation
- Explore code files to understand implementation
- Try different models and configurations

## 💡 Pro Tips

1. **Start small**: Use `--epochs 1` and `--batch-size 4` for quick tests
2. **Monitor GPU**: Use `nvidia-smi` to check memory usage
3. **Save checkpoints**: Training saves every 500 steps
4. **Use validation**: Monitor validation loss to prevent overfitting
5. **Experiment**: Try different LoRA ranks (r=8, 16, 32)

## 🆘 Getting Help

- Check error messages carefully
- Review configuration in `config.py`
- See examples in each script's `__main__` block
- Refer to library documentation (Transformers, PEFT, TRL)

---

Happy fine-tuning! 🚀
