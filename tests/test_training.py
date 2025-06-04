#!/usr/bin/env python3

import os
import sys
import torch

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training import DS1000FineTuner

def test_finetuning_dry_run():
    """Test fine-tuning setup without actual training"""
    print("=== Testing Fine-tuning Setup ===")
    
    try:
        fine_tuner = DS1000FineTuner()
        print("✓ Fine-tuner initialized successfully")
        
        data = fine_tuner.load_training_data()
        print(f"✓ Loaded {len(data)} training examples")
        
        train_dataset, val_dataset = fine_tuner.prepare_dataset(data[:20])
        print(f"✓ Prepared datasets: train={len(train_dataset)}, val={len(val_dataset)}")
        
        training_args = fine_tuner.setup_training_args(
            output_dir="./test_model",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=2,
            save_steps=1000,
            eval_steps=1000
        )
        print("✓ Training arguments configured")
        
        print("✓ Fine-tuning setup test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Fine-tuning setup failed: {e}")
        return False

def test_quick_training():
    """Test actual training with minimal steps"""
    print("\n=== Testing Quick Training ===")
    
    try:
        fine_tuner = DS1000FineTuner()
        
        trainer = fine_tuner.train(
            output_dir="./test_model_quick",
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=3,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=1,
            eval_strategy="no",
            load_best_model_at_end=False
        )
        
        print("✓ Quick training completed successfully!")
        
        # Check if model was saved
        if os.path.exists("./test_model_quick/pytorch_model.bin"):
            print("✓ Model checkpoint saved")
        else:
            print("⚠ Model checkpoint not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Quick training failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    import shutil
    test_dirs = ["./test_model", "./test_model_quick"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"Cleaned up {test_dir}")

def main():
    print("Testing Fine-tuning Functionality")
    print("=" * 40)
    
    setup_success = test_finetuning_dry_run()
    
    if setup_success:
        training_success = test_quick_training()
    else:
        print("Skipping training test due to setup failure")
        training_success = False
    
    cleanup()
    
    if setup_success and training_success:
        print("\n✅ All fine-tuning tests passed!")
    else:
        print("\n❌ Some fine-tuning tests failed")

if __name__ == "__main__":
    main() 