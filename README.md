# Specialty-Tuner-Light

A lightweight toolkit for fine-tuning code generation models using the DS-1000 dataset.

## Features

- **DS-1000 Dataset Processing**: Download, preprocess, and prepare the DS-1000 dataset for fine-tuning
- **Model Evaluation**: Comprehensive evaluation framework with before/during/after training metrics
- **DeepSeek Integration**: Ready-to-use setup with DeepSeek Coder models
- **MLX Support**: Apple Silicon optimized inference with MLX

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download and preprocess DS-1000**:
   ```bash
   python -m src.dataset
   ```

3. **Evaluate your model**:
   ```bash
   python -m src.evaluation --max_samples 100
   ```

4. **Run tests**:
   ```bash
   python tests/test_suite.py
   ```

## Project Structure

```
src/
├── dataset.py          # Dataset processing and preparation
├── evaluation.py       # Model evaluation framework
├── training.py         # Fine-tuning functionality
└── models.py           # Model setup (PyTorch + MLX)

scripts/
└── compare.py          # Model comparison utilities

tests/
├── test_training.py    # Training functionality tests
└── test_suite.py       # Comprehensive test suite

docs/
├── README.md           # Detailed documentation
├── finetuning.md       # Fine-tuning guide
└── testing.md          # Testing documentation

data/                   # Dataset storage
requirements.txt        # Dependencies
```

## Usage Examples

### Basic Evaluation
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
metrics, results = evaluator.run_baseline_evaluation(max_samples=50)
```

### Dataset Processing
```python
from src.dataset import DS1000Processor

processor = DS1000Processor()
dataset = processor.download_dataset()
processor.preprocess_for_finetuning(dataset)
```

### Model Comparison
```bash
python scripts/compare.py --base_model deepseek-ai/deepseek-coder-1.3b-instruct --finetuned_model ./finetuned_model
```

### Fine-tuning
```bash
python -m src.training --epochs 3 --batch_size 4 --output_dir ./my_model
```

## Command Line Usage

```bash
# Basic evaluation
python -m src.evaluation

# Custom model evaluation
python -m src.evaluation --model "your-model-name" --max_samples 100

# Compare multiple models
python -m src.evaluation --mode compare --models model1 model2 model3

# Fine-tune a model
python -m src.training --model deepseek-ai/deepseek-coder-1.3b-instruct --epochs 3
```

## Output Files

The toolkit creates:
- `data/processed/ds1000_finetuning.jsonl`: Training data in JSONL format
- `data/processed/ds1000_train.json`: Training split
- `data/processed/ds1000_validation.json`: Validation split  
- `data/processed/ds1000_test.json`: Test split
- `data/processed/evaluation_results.json`: Evaluation metrics and results

## Documentation

- [Detailed Usage Guide](docs/README.md)
- [Fine-tuning Guide](docs/finetuning.md)
- [Testing Documentation](docs/testing.md) 