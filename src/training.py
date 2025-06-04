#!/usr/bin/env python3
"""
Model fine-tuning functionality for code generation models.

This module provides tools for fine-tuning pre-trained language models on the
DS-1000 dataset using the Hugging Face Transformers library. Features include:
- Automatic data loading and preparation
- Device-aware training configuration
- Support for various training parameters
- Model saving and checkpointing
"""

import os
import json
import torch
import argparse
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from .dataset import DS1000Processor

class DS1000FineTuner:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct", max_length=1024):
        self.model_name = model_name
        self.max_length = max_length
        self.device = self._get_device()
        
        print(f"Initializing fine-tuner with {model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "mps":
            self.model = self.model.to(self.device)
    
    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def load_training_data(self, data_path="./data/processed"):
        """Load and prepare training data"""
        train_file = os.path.join(data_path, "ds1000_finetuning.jsonl")
        
        if not os.path.exists(train_file):
            print(f"Training data not found at {train_file}")
            print("Running data preparation...")
            processor = DS1000Processor()
            dataset = processor.download_dataset()
            processor.preprocess_for_finetuning(dataset)
        
        print(f"Loading training data from {train_file}")
        data = []
        with open(train_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        print(f"Loaded {len(data)} training examples")
        return data
    
    def prepare_dataset(self, data, train_split=0.9):
        """Prepare dataset for training"""
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        def format_training_text(example):
            return f"{example['prompt']}\n{example['completion']}"
        
        train_texts = [format_training_text(ex) for ex in train_data]
        val_texts = [format_training_text(ex) for ex in val_data]
        
        train_dataset = Dataset.from_dict({"text": train_texts})
        val_dataset = Dataset.from_dict({"text": val_texts})
        
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_training_args(self, output_dir="./finetuned_model", **kwargs):
        """Setup training arguments"""
        default_args = {
            "output_dir": output_dir,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "eval_steps": 100,
            "save_steps": 500,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "report_to": None,
            "remove_unused_columns": False,
            "dataloader_pin_memory": False,
        }
        
        # Device-specific adjustments
        if self.device == "mps":
            default_args["dataloader_pin_memory"] = False
            default_args["fp16"] = False
        elif self.device == "cuda":
            default_args["fp16"] = True
        
        # Override with user arguments
        default_args.update(kwargs)
        
        return TrainingArguments(**default_args)
    
    def train(self, output_dir="./finetuned_model", **training_kwargs):
        """Fine-tune the model"""
        print("Preparing training data...")
        data = self.load_training_data()
        train_dataset, val_dataset = self.prepare_dataset(data)
        
        print("Setting up training configuration...")
        training_args = self.setup_training_args(output_dir, **training_kwargs)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training...")
        print(f"Training for {training_args.num_train_epochs} epochs")
        print(f"Batch size: {training_args.per_device_train_batch_size}")
        print(f"Learning rate: {training_args.learning_rate}")
        
        trainer.train()
        
        print("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune model on DS-1000 dataset")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct", help="Base model to fine-tune")
    parser.add_argument("--output_dir", default="./finetuned_model", help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    
    args = parser.parse_args()
    
    fine_tuner = DS1000FineTuner(
        model_name=args.model,
        max_length=args.max_length
    )
    
    training_kwargs = {
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }
    
    trainer = fine_tuner.train(args.output_dir, **training_kwargs)
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main() 