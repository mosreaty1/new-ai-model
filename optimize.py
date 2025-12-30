"""
Model optimization script for inference.
Phase 3: Optimization
"""
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import time
import numpy as np


def benchmark_model(model, tokenizer, sample_texts, num_runs=10):
    """Benchmark model inference speed."""
    device = next(model.parameters()).device

    latencies = []
    for _ in range(num_runs):
        inputs = tokenizer(
            sample_texts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(device)

        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=64)
        end = time.time()

        latencies.append((end - start) / len(sample_texts))

    return {
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies)
    }


def get_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 ** 2
    return size_mb


def quantize_model(model, tokenizer, output_dir):
    """Quantize model to int8."""
    print("Quantizing model to int8...")

    # Convert to int8
    model = model.to(torch.int8)

    # Save quantized model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Quantized model saved to {output_dir}")


def main(args):
    """Main optimization function."""
    print("=" * 80)
    print("Model Optimization for Inference")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base_model, args.model_path)

    # Merge LoRA weights for faster inference
    if args.merge_lora:
        print("\nMerging LoRA weights...")
        model = model.merge_and_unload()

    # Move to device
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"\nUsing device: {device}")
    model = model.to(device)
    model.eval()

    # Calculate model size
    print("\n" + "=" * 80)
    print("Model Size Analysis")
    print("=" * 80)

    size_mb = get_model_size(model)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Benchmark
    if args.benchmark:
        print("\n" + "=" * 80)
        print("Benchmarking Inference Speed")
        print("=" * 80)

        sample_texts = [
            "Extract the positive sentiment phrase from: I really love this product!",
            "Extract the negative sentiment phrase from: This is terrible and I hate it.",
            "Extract the neutral sentiment phrase from: It is what it is.",
        ]

        print(f"Running {args.num_runs} iterations...")
        metrics = benchmark_model(model, tokenizer, sample_texts, num_runs=args.num_runs)

        print(f"\nLatency Statistics (per sample):")
        print(f"  Mean: {metrics['mean_latency']*1000:.2f} ms")
        print(f"  Std:  {metrics['std_latency']*1000:.2f} ms")
        print(f"  Min:  {metrics['min_latency']*1000:.2f} ms")
        print(f"  Max:  {metrics['max_latency']*1000:.2f} ms")

    # Save optimized model
    if args.output_dir:
        print("\n" + "=" * 80)
        print("Saving Optimized Model")
        print("=" * 80)

        print(f"Saving to {args.output_dir}...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        print("Optimized model saved successfully!")

    # Optimization recommendations
    print("\n" + "=" * 80)
    print("Optimization Recommendations")
    print("=" * 80)

    print("""
    1. **Merge LoRA weights**: Use --merge-lora for faster inference (no adapter overhead)
    2. **Use FP16/BF16**: Enable half-precision for 2x speedup with minimal quality loss
    3. **Quantization**: Use int8 or int4 quantization for smaller model size
    4. **Batch inference**: Process multiple samples together for better throughput
    5. **ONNX export**: Convert to ONNX format for optimized runtime
    6. **KV-cache optimization**: Enable key-value caching for sequential generation
    7. **Model compilation**: Use torch.compile() for additional speedup (PyTorch 2.0+)
    """)

    print("=" * 80)
    print("Optimization complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize model for inference")

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/flan-t5-base",
        help="Base model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for optimized model"
    )
    parser.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA weights into base model"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmarks"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference"
    )

    args = parser.parse_args()
    main(args)
