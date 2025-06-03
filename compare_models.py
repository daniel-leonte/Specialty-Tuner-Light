#!/usr/bin/env python3

import argparse
import torch
from deepseek_setup import DeepSeekCoder
from ds1000_processor import DS1000Processor, DS1000Evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelComparator:
    def __init__(self):
        self.processor = DS1000Processor()
        self.evaluator = DS1000Evaluator(self.processor)
        
    def load_model(self, model_path, is_local=False):
        """Load model from path (local or HuggingFace)"""
        if is_local:
            print(f"Loading local model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            print(f"Loading HuggingFace model: {model_path}")
            coder = DeepSeekCoder(model_path)
            model = coder.model
            tokenizer = coder.tokenizer
            
        return model, tokenizer
    
    def evaluate_model(self, model, tokenizer, test_data, model_name, max_samples=50):
        """Evaluate a single model"""
        print(f"\n=== Evaluating {model_name} ===")
        
        metrics, results = self.evaluator.evaluate_model(
            model, tokenizer, test_data, max_samples=max_samples
        )
        
        print(f"Results for {model_name}:")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        print(f"  Total Examples: {metrics['total_examples']}")
        print(f"  Exact Matches: {metrics['exact_matches']}")
        
        return metrics, results
    
    def compare_models(self, base_model_path, finetuned_model_path, max_samples=50):
        """Compare base model vs fine-tuned model"""
        print("Loading test data...")
        
        # Load test data
        dataset = self.processor.download_dataset()
        splits = self.processor.create_splits(dataset)
        test_data = splits["test"][:max_samples]
        
        print(f"Comparing models on {len(test_data)} test examples")
        
        # Load models
        base_model, base_tokenizer = self.load_model(base_model_path, is_local=False)
        ft_model, ft_tokenizer = self.load_model(finetuned_model_path, is_local=True)
        
        # Evaluate both models
        base_metrics, base_results = self.evaluate_model(
            base_model, base_tokenizer, test_data, "Base Model", max_samples
        )
        
        ft_metrics, ft_results = self.evaluate_model(
            ft_model, ft_tokenizer, test_data, "Fine-tuned Model", max_samples
        )
        
        # Compare results
        self.print_comparison(base_metrics, ft_metrics)
        
        return {
            "base": {"metrics": base_metrics, "results": base_results},
            "finetuned": {"metrics": ft_metrics, "results": ft_results}
        }
    
    def print_comparison(self, base_metrics, ft_metrics):
        """Print detailed comparison"""
        print("\n" + "="*50)
        print("MODEL COMPARISON RESULTS")
        print("="*50)
        
        base_acc = base_metrics['exact_match_accuracy']
        ft_acc = ft_metrics['exact_match_accuracy']
        improvement = ft_acc - base_acc
        improvement_pct = (improvement / base_acc * 100) if base_acc > 0 else 0
        
        print(f"Base Model Accuracy:       {base_acc:.3f}")
        print(f"Fine-tuned Model Accuracy: {ft_acc:.3f}")
        print(f"Improvement:               {improvement:+.3f} ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print("🎉 Fine-tuning improved performance!")
        elif improvement < 0:
            print("⚠️  Fine-tuning decreased performance")
        else:
            print("➡️  No change in performance")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Compare base model vs fine-tuned model")
    parser.add_argument("--base_model", default="deepseek-ai/deepseek-coder-1.3b-instruct", 
                       help="Base model name or path")
    parser.add_argument("--finetuned_model", default="./finetuned_model", 
                       help="Path to fine-tuned model")
    parser.add_argument("--max_samples", type=int, default=50, 
                       help="Maximum test samples to evaluate")
    
    args = parser.parse_args()
    
    comparator = ModelComparator()
    results = comparator.compare_models(
        args.base_model, 
        args.finetuned_model, 
        args.max_samples
    )
    
    print(f"\nComparison complete! Results saved to evaluation_results.json")

if __name__ == "__main__":
    main() 