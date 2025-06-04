# DS-1000 Fine-tuning Guide

This guide explains how to fine-tune DeepSeek models on the DS-1000 dataset for improved code generation performance.

## Quick Start

### 1. Setup Environment
```bash
venvup  # Activate virtual environment
pip install -r requirements.txt
```

### 2. Test Fine-tuning Setup
```bash
python test_finetuning.py
```
This validates that all components work correctly.

## Fine-tuning Process

### Basic Fine-tuning
```bash
python finetune_ds1000.py
```

### Custom Fine-tuning Options
```bash
python finetune_ds1000.py \
    --model deepseek-ai/deepseek-coder-1.3b-instruct \
    --output_dir ./my_finetuned_model \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --max_length 1024 \
    --gradient_accumulation_steps 4
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `deepseek-ai/deepseek-coder-1.3b-instruct` | Base model to fine-tune |
| `--output_dir` | `./finetuned_model` | Output directory for fine-tuned model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Training batch size per device |
| `--learning_rate` | `5e-5` | Learning rate |
| `--max_length` | `1024` | Maximum sequence length |
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |

## Dataset Format

The fine-tuning uses your DS-1000 dataset with the "Problem:" format, which is **perfect for fine-tuning** because:

1. **Consistent Structure**: Each example follows a clear pattern:
   ```
   # Problem:
   [Problem description with context]
   
   # Solution:
   [Reference code solution]
   ```

2. **Instruction Format**: The "Problem:" prefix creates a natural instruction-following format that helps the model understand the task structure.

3. **Original DS-1000 Format**: This preserves the authentic dataset format, ensuring compatibility with evaluation benchmarks.

## Evaluation and Comparison

### Compare Fine-tuned vs Base Model
```bash
python compare_models.py \
    --base_model deepseek-ai/deepseek-coder-1.3b-instruct \
    --finetuned_model ./finetuned_model \
    --max_samples 100
```

### Evaluate Fine-tuned Model Only
```bash
python evaluate_with_ds1000.py \
    --mode custom \
    --test_file "./ds1000_data/processed/ds1000_test.json" \
    --max_samples 50
```

## Training Configuration

### Device-Specific Settings

**Apple Silicon (MPS)**:
- Automatically disables FP16 (not supported)
- Uses float32 precision
- Optimized memory settings

**CUDA GPU**:
- Enables FP16 for faster training
- Uses device_map="auto" for multi-GPU

**CPU**:
- Uses float32 precision
- Reduced batch sizes recommended

### Memory Optimization

For limited memory:
```bash
python finetune_ds1000.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_length 512
```

For more memory:
```bash
python finetune_ds1000.py \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_length 1024
```

## Expected Results

### Training Progress
- **Loss**: Should decrease from ~1.0 to <0.5
- **Training Time**: ~2-4 hours for 3 epochs (depending on hardware)
- **Memory Usage**: ~4-8GB GPU memory

### Performance Improvements
- **Baseline Model**: ~10-20% accuracy on DS-1000
- **Fine-tuned Model**: Expected 20-40% improvement
- **Best Case**: 30-60% accuracy on DS-1000 tasks

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `--batch_size` to 1
   - Increase `--gradient_accumulation_steps`
   - Reduce `--max_length`

2. **Slow Training**:
   - Increase `--batch_size` if memory allows
   - Use GPU instead of CPU
   - Reduce dataset size for testing

3. **Poor Performance**:
   - Increase training epochs
   - Adjust learning rate (try 1e-5 or 1e-4)
   - Check data quality

### Validation

Always run the test suite before full training:
```bash
python test_finetuning.py
```

## File Structure

After fine-tuning, you'll have:
```
./finetuned_model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## Next Steps

1. **Fine-tune your model**: `python finetune_ds1000.py`
2. **Compare performance**: `python compare_models.py`
3. **Use for inference**: Load the model from `./finetuned_model`

The fine-tuned model can be used anywhere you'd use the base model, with improved performance on DS-1000 style coding tasks. 