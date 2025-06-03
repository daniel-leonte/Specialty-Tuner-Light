import torch
import json
import argparse
from deepseek_setup import DeepSeekCoder
from ds1000_processor import DS1000Processor, DS1000Evaluator

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_info():
    """Get device information"""
    device = get_device()
    info = {"device": device}
    
    if device == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    
    return info

class ModelEvaluator:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        self.coder = DeepSeekCoder(model_name)
        self.processor = DS1000Processor()
        self.evaluator = DS1000Evaluator(self.processor)
    
    def run_baseline_evaluation(self, max_samples=50):
        """Run baseline evaluation"""
        print("=== Baseline Evaluation ===")
        
        device_info = get_device_info()
        print(f"Device: {device_info['device']}")
        if 'cuda_device_name' in device_info:
            print(f"GPU: {device_info['cuda_device_name']} ({device_info['cuda_memory_total']}GB)")
        
        dataset = self.processor.download_dataset()
        splits = self.processor.create_splits(dataset)
        
        return self.evaluator.evaluate_model(
            self.coder.model, 
            self.coder.tokenizer, 
            splits["test"], 
            max_samples=max_samples
        )
    
    def run_custom_evaluation(self, test_file, max_samples=50):
        """Run evaluation on custom test file"""
        print(f"=== Custom Evaluation on {test_file} ===")
        
        return self.evaluator.evaluate_model(
            self.coder.model,
            self.coder.tokenizer,
            test_file,
            max_samples=max_samples
        )

def compare_models(model_names, max_samples=20):
    """Compare multiple models on DS-1000"""
    print("=== Model Comparison ===")
    print(f"Using device: {get_device()}")
    
    processor = DS1000Processor()
    evaluator = DS1000Evaluator(processor)
    
    dataset = processor.download_dataset()
    splits = processor.create_splits(dataset)
    test_data = splits["test"][:max_samples]
    
    results = {}
    
    for model_name in model_names:
        print(f"\nEvaluating {model_name}...")
        
        coder = DeepSeekCoder(model_name)
        metrics, _ = evaluator.evaluate_model(
            coder.model,
            coder.tokenizer,
            test_data,
            max_samples=max_samples
        )
        
        results[model_name] = metrics
        
        del coder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n=== Comparison Results ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics['exact_match_accuracy']:.3f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate models on DS-1000 dataset")
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-1.3b-instruct", help="Model to evaluate")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum samples to evaluate")
    parser.add_argument("--mode", choices=["baseline", "custom", "compare"], default="baseline", help="Evaluation mode")
    parser.add_argument("--test_file", help="Custom test file path")
    parser.add_argument("--models", nargs="+", help="Models to compare")
    
    args = parser.parse_args()
    
    device_info = get_device_info()
    print(f"System: {device_info}")
    
    if args.mode == "baseline":
        evaluator = ModelEvaluator(args.model)
        metrics, results = evaluator.run_baseline_evaluation(args.max_samples)
        
    elif args.mode == "custom" and args.test_file:
        evaluator = ModelEvaluator(args.model)
        metrics, results = evaluator.run_custom_evaluation(args.test_file, args.max_samples)
        
    elif args.mode == "compare" and args.models:
        results = compare_models(args.models, args.max_samples)
    
    else:
        print("Invalid arguments. Use --help for usage information.")

if __name__ == "__main__":
    main() 