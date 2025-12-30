# Makefile for AIE417 Tweet Sentiment Extraction Project

.PHONY: help install setup train evaluate infer optimize deploy clean test

# Default target
help:
	@echo "AIE417 Tweet Sentiment Extraction - Available Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install Python dependencies"
	@echo "  make setup         - Complete project setup"
	@echo ""
	@echo "Phase 1 - Data Analysis:"
	@echo "  make analyze       - Analyze dataset statistics"
	@echo ""
	@echo "Phase 2 - Training:"
	@echo "  make train         - Train model with PEFT/LoRA (full)"
	@echo "  make train-quick   - Quick training (1 epoch for testing)"
	@echo "  make train-qlora   - Train with quantization (QLoRA)"
	@echo "  make align         - Run DPO alignment (optional)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make evaluate      - Evaluate model performance"
	@echo "  make eval-full     - Full evaluation with error analysis"
	@echo ""
	@echo "Inference:"
	@echo "  make infer         - Generate predictions on test set"
	@echo ""
	@echo "Phase 3 - Optimization & Deployment:"
	@echo "  make optimize      - Optimize model for inference"
	@echo "  make app           - Run Streamlit app locally"
	@echo "  make deploy        - Deploy to Hugging Face"
	@echo ""
	@echo "Utilities:"
	@echo "  make test          - Run basic tests"
	@echo "  make clean         - Clean generated files"
	@echo "  make clean-all     - Clean everything including models"
	@echo ""

# Installation and setup
install:
	pip install -r requirements.txt

setup: install
	@echo "Creating necessary directories..."
	@mkdir -p models
	@mkdir -p evaluation
	@mkdir -p outputs
	@echo "Setup complete!"
	@echo "Note: Copy .env.example to .env and configure if needed"

# Phase 1: Data Analysis
analyze:
	python data_preprocessing.py

# Phase 2: Training
train:
	python train.py \
		--epochs 3 \
		--batch-size 8 \
		--learning-rate 3e-4

train-quick:
	python train.py \
		--epochs 1 \
		--batch-size 4

train-qlora:
	python train.py \
		--use-quantization \
		--epochs 3 \
		--batch-size 4

align:
	python alignment.py \
		--model-path ./models/flan-t5-sentiment-extraction \
		--output-dir ./models/flan-t5-aligned \
		--num-samples 1000

# Evaluation
evaluate:
	python evaluate.py \
		--model-path ./models/flan-t5-sentiment-extraction \
		--visualize

eval-full:
	python evaluate.py \
		--model-path ./models/flan-t5-sentiment-extraction \
		--error-analysis \
		--visualize \
		--save-predictions \
		--top-k-errors 20

# Inference
infer:
	python inference.py \
		--model-path ./models/flan-t5-sentiment-extraction \
		--test-file test.csv \
		--output-file submission.csv

infer-aligned:
	python inference.py \
		--model-path ./models/flan-t5-aligned \
		--test-file test.csv \
		--output-file submission_aligned.csv

# Phase 3: Optimization
optimize:
	python optimize.py \
		--model-path ./models/flan-t5-sentiment-extraction \
		--merge-lora \
		--benchmark \
		--output-dir ./models/flan-t5-optimized

# Deployment
app:
	streamlit run app.py

deploy:
	@echo "Make sure to set HF_TOKEN environment variable!"
	python deploy_hf.py \
		--model-path ./models/flan-t5-sentiment-extraction \
		--repo-name $(USER)/flan-t5-sentiment-extraction

# Testing
test:
	@echo "Running basic tests..."
	python -c "from config import ProjectConfig; print('✓ Config loaded')"
	python -c "from data_preprocessing import SentimentDataPreprocessor; print('✓ Preprocessor loaded')"
	python -c "from model_trainer import SentimentExtractionTrainer; print('✓ Trainer loaded')"
	@echo "All imports successful!"

# Cleaning
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f submission.csv submission_aligned.csv
	@echo "Cleanup complete!"

clean-all: clean
	@echo "WARNING: This will delete all trained models!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf models/*; \
		rm -rf evaluation/*; \
		rm -rf outputs/*; \
		echo "All models and outputs deleted!"; \
	fi

# Development helpers
dev-setup: setup
	pip install jupyter ipython black flake8 pytest

format:
	black *.py

lint:
	flake8 *.py --max-line-length=120

# Complete workflow shortcuts
workflow-basic: analyze train evaluate infer
	@echo "Basic workflow complete!"

workflow-full: analyze train align evaluate optimize infer
	@echo "Full workflow complete!"

# Help for specific phases
help-phase1:
	@echo "Phase 1: Project Proposal"
	@echo "========================="
	@echo "1. Run: make analyze"
	@echo "2. Review dataset statistics"
	@echo "3. Read: PROJECT_REPORT.md (Phase 1 section)"

help-phase2:
	@echo "Phase 2: Fine-tuning and Alignment"
	@echo "===================================="
	@echo "1. Run: make train"
	@echo "2. Optional: make align"
	@echo "3. Run: make evaluate"

help-phase3:
	@echo "Phase 3: Optimization and Deployment"
	@echo "====================================="
	@echo "1. Run: make optimize"
	@echo "2. Run: make app (test locally)"
	@echo "3. Run: make deploy (publish to HF)"

# Presentation commands
presentation-prep:
	@echo "Preparing for final presentation..."
	@echo "1. Running baseline vs fine-tuned comparison..."
	python demo_comparison.py
	@echo "2. Generating evaluation visualizations..."
	make eval-full
	@echo "3. Creating sample submission..."
	make infer
	@echo ""
	@echo "✓ Presentation materials ready!"
	@echo "  - comparison.png (baseline vs fine-tuned)"
	@echo "  - evaluation/evaluation_results.png"
	@echo "  - submission.csv"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Review PRESENTATION_GUIDE.md"
	@echo "  2. Start Streamlit: make demo"
	@echo "  3. Practice your presentation!"

demo:
	@echo "Starting Streamlit demo app..."
	@echo "Access at: http://localhost:8501"
	streamlit run app.py

demo-comparison:
	@echo "Running baseline vs fine-tuned comparison..."
	python demo_comparison.py
	@echo "Results saved to comparison.png"

kaggle-submit:
	@echo "Generating Kaggle submission..."
	make infer
	@echo "Upload submission.csv to Kaggle!"
