# Specialty-Tuner-Light

A lightweight toolkit for fine-tuning code generation models using the DS-1000 dataset.

## Features

- 🔄 **DS-1000 Dataset Processing**: Download, preprocess, and prepare the DS-1000 dataset for fine-tuning
- 📊 **Model Evaluation**: Comprehensive evaluation framework with before/during/after training metrics
- 🚀 **DeepSeek Integration**: Ready-to-use setup with DeepSeek Coder models
- 📱 **MLX Support**: Apple Silicon optimized inference with MLX

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the demo**:
   ```bash
   python usage_example.py
   ```

3. **Download and preprocess DS-1000**:
   ```bash
   python ds1000_processor.py
   ```

4. **Evaluate your model**:
   ```bash
   python evaluate_with_ds1000.py --max_samples 100
   ```

## Dataset Processing

The `DS1000Processor` class handles:
- Downloading DS-1000 from Hugging Face or GitHub
- Creating train/validation/test splits
- Formatting data for fine-tuning (JSONL/JSON output)
- Dataset statistics and analysis

## Model Evaluation

Use the `DS1000Evaluator` for:
- Baseline evaluation before fine-tuning
- Custom evaluation on specific test sets
- Multi-model comparison
- Tracking improvements during training

### Example Usage

```python
from ds1000_processor import DS1000Processor
from evaluate_with_ds1000 import DS1000Evaluator

# Process dataset
processor = DS1000Processor()
dataset = processor.download_dataset()
processor.preprocess_for_finetuning(dataset)

# Evaluate model
evaluator = DS1000Evaluator()
metrics, results = evaluator.run_baseline_evaluation(max_samples=50)
```

## Command Line Usage

```bash
# Basic evaluation
python evaluate_with_ds1000.py

# Custom model evaluation
python evaluate_with_ds1000.py --model "your-model-name" --max_samples 100

# Compare multiple models
python evaluate_with_ds1000.py --mode compare --models model1 model2 model3

# Evaluate on custom test file
python evaluate_with_ds1000.py --mode custom --test_file path/to/test.json
```

## File Structure

- `ds1000_processor.py`: Main dataset processing functionality
- `evaluate_with_ds1000.py`: Model evaluation framework
- `deepseek_setup.py`: DeepSeek model setup (PyTorch)
- `deepseek_mlx_setup.py`: DeepSeek model setup (MLX/Apple Silicon)
- `usage_example.py`: Demo and example usage

## Output Files

The processor creates:
- `ds1000_finetuning.jsonl`: Training data in JSONL format
- `ds1000_train.json`: Training split
- `ds1000_validation.json`: Validation split  
- `ds1000_test.json`: Test split
- `evaluation_results.json`: Evaluation metrics and results
