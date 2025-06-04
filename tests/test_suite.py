#!/usr/bin/env python3
"""
Test Suite for DS-1000 Evaluation System

This script tests all major components:
1. Dataset processing
2. Model setup 
3. Evaluation functionality
"""

import os
import json
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        from src.models import DeepSeekCoder
        from src.dataset import DS1000Processor, DS1000Evaluator
        from src.evaluation import ModelEvaluator, get_device
        import torch
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_device_detection():
    """Test device detection functionality"""
    print("Testing device detection...")
    try:
        from src.evaluation import get_device, get_device_info
        device = get_device()
        info = get_device_info()
        print(f"✓ Device detected: {device}")
        print(f"✓ Device info: {info}")
        return True
    except Exception as e:
        print(f"✗ Device detection failed: {e}")
        return False

def test_dataset_processor():
    """Test dataset processing functionality"""
    print("Testing dataset processor...")
    try:
        from src.dataset import DS1000Processor
        
        processor = DS1000Processor()
        dataset = processor.download_dataset()
        stats = processor.get_statistics(dataset)
        
        print(f"✓ Dataset downloaded: {stats['total_examples']} examples")
        
        splits = processor.create_splits(dataset)
        print(f"✓ Splits created: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset processing failed: {e}")
        return False

def test_model_setup():
    """Test model setup functionality"""
    print("Testing model setup...")
    try:
        from src.models import DeepSeekCoder
        
        coder = DeepSeekCoder()
        test_prompt = "def hello():"
        result = coder.generate(test_prompt, max_length=50)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Generated code (truncated): {result[:100]}...")
        
        return True
    except Exception as e:
        print(f"✗ Model setup failed: {e}")
        return False

def test_evaluation_pipeline():
    """Test the complete evaluation pipeline"""
    print("Testing evaluation pipeline...")
    try:
        from src.evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        metrics, results = evaluator.run_baseline_evaluation(max_samples=2)
        
        print(f"✓ Evaluation completed")
        print(f"✓ Metrics: {metrics}")
        print(f"✓ Results: {len(results)} examples evaluated")
        
        return True
    except Exception as e:
        print(f"✗ Evaluation pipeline failed: {e}")
        return False

def test_file_outputs():
    """Test that expected output files exist"""
    print("Testing file outputs...")
    
    expected_files = [
        "data/processed/ds1000_train.json",
        "data/processed/ds1000_validation.json", 
        "data/processed/ds1000_test.json",
        "data/processed/ds1000_finetuning.jsonl",
        "data/processed/evaluation_results.json"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} exists ({size} bytes)")
        else:
            print(f"✗ {file_path} missing")
            all_exist = False
    
    return all_exist

def run_full_test_suite():
    """Run the complete test suite"""
    print("=" * 60)
    print("DS-1000 Evaluation System Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Detection", test_device_detection),
        ("Dataset Processing", test_dataset_processor),
        ("Model Setup", test_model_setup),
        ("Evaluation Pipeline", test_evaluation_pipeline),
        ("File Outputs", test_file_outputs)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your DS-1000 evaluation system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_test_suite()
    sys.exit(0 if success else 1) 