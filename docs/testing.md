# DS-1000 Evaluation System Testing Guide

This guide explains how to test your DS-1000 evaluation system components.

## Quick Start

### 1. Setup Environment
```bash
venvup  # Activate virtual environment
pip install -r requirements.txt
```

### 2. Run Complete Test Suite
```bash
python test_suite.py
```
This runs all tests and validates the complete system.

## Individual Component Testing

### Test Dataset Processing
```bash
python ds1000_processor.py
```
- Downloads DS-1000 dataset
- Creates train/val/test splits
- Generates preprocessed files

### Test Model Setup
```bash
python deepseek_setup.py
```
- Loads DeepSeek model
- Tests code generation

### Test Evaluation Pipeline

#### Baseline Evaluation (small sample)
```bash
python evaluate_with_ds1000.py --mode baseline --max_samples 5
```

#### Custom Test File Evaluation
```bash
python evaluate_with_ds1000.py --mode custom --test_file "./ds1000_data/processed/ds1000_test.json" --max_samples 3
```

#### Model Comparison
```bash
python evaluate_with_ds1000.py --mode compare --models "deepseek-ai/deepseek-coder-1.3b-instruct" --max_samples 5
```

## Command Line Options

### evaluate_with_ds1000.py Options:
- `--model`: Model to evaluate (default: deepseek-ai/deepseek-coder-1.3b-instruct)
- `--max_samples`: Maximum samples to evaluate (default: 50)
- `--mode`: Evaluation mode (baseline, custom, compare)
- `--test_file`: Custom test file path
- `--models`: List of models to compare

### Examples:
```bash
# Quick test with 3 samples
python evaluate_with_ds1000.py --mode baseline --max_samples 3

# Full baseline evaluation
python evaluate_with_ds1000.py --mode baseline --max_samples 100

# Custom evaluation
python evaluate_with_ds1000.py --mode custom --test_file "my_test.json" --max_samples 10

# Compare models
python evaluate_with_ds1000.py --mode compare --models "model1" "model2" --max_samples 20
```

## Generated Files

After running the system, you'll find these files in `ds1000_data/processed/`:

- `ds1000_train.json`: Training split (800 examples)
- `ds1000_validation.json`: Validation split (100 examples)
- `ds1000_test.json`: Test split (100 examples)
- `ds1000_finetuning.jsonl`: Fine-tuning format data (1000 examples)
- `evaluation_results.json`: Latest evaluation results and metrics

## Understanding Results

The evaluation produces:
- **Exact Match Accuracy**: Percentage of exactly matching generated vs reference code
- **Individual Results**: Per-example generated code, reference code, and match status
- **Metrics**: Total examples, exact matches, accuracy

## Device Support

The system automatically detects and uses:
- **CUDA**: NVIDIA GPUs (if available)
- **MPS**: Apple Silicon GPUs (if available)
- **CPU**: Fallback option

## Troubleshooting

### Common Issues:

1. **Memory Issues**: Reduce `--max_samples` parameter
2. **Model Loading**: Ensure sufficient disk space and memory
3. **Dataset Download**: Check internet connection

### Performance Tips:

- Use smaller sample sizes for testing (`--max_samples 5-10`)
- Monitor GPU memory usage with large models
- Use CPU if GPU memory is insufficient

## System Requirements

- Python 3.8+
- 8GB+ RAM (16GB+ recommended for larger models)
- 10GB+ free disk space
- GPU optional but recommended for faster inference 