import os
import json
from datasets import load_dataset
from tqdm import tqdm
import torch

class DS1000Processor:
    def __init__(self, data_dir="./ds1000_data"):
        self.data_dir = data_dir
        self.processed_data_path = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_data_path, exist_ok=True)
        
    def download_dataset(self):
        """Download DS-1000 dataset from Hugging Face"""
        print("Downloading DS-1000 dataset...")
        try:
            return load_dataset("xlangai/DS-1000")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise
    
    def preprocess_for_finetuning(self, dataset, output_format="jsonl"):
        """Preprocess dataset for fine-tuning"""
        print("Preprocessing dataset for fine-tuning...")
        processed_data = []
        
        for split in dataset:
            for example in tqdm(dataset[split], desc=f"Processing {split}"):
                processed_example = self._format_example(example)
                processed_data.append(processed_example)
        
        output_file = os.path.join(self.processed_data_path, f"ds1000_finetuning.{output_format}")
        self._save_data(processed_data, output_file, output_format)
        
        print(f"Processed {len(processed_data)} examples saved to {output_file}")
        return processed_data
    
    def _format_example(self, example):
        """Format individual example for training"""
        return {
            "prompt": self._create_prompt(example),
            "completion": example.get("reference_code", ""),
            "metadata": {
                "lib": example.get("lib", ""),
                "difficulty": example.get("difficulty", ""),
                "problem_id": example.get("problem_id", "")
            }
        }
    
    def _create_prompt(self, example):
        """Create training prompt from example"""
        parts = []
        
        if "import" in example:
            parts.append(f"# Imports:\n{example['import']}")
        
        if "prompt" in example:
            problem_text = example['prompt']
            if problem_text.startswith("Problem:"):
                problem_text = problem_text[8:].strip()
            parts.append(f"# Problem:\n{problem_text}")
        
        if "context" in example:
            parts.append(f"# Context:\n{example['context']}")
        
        parts.append("# Solution:")
        return "\n\n".join(parts)
    
    def create_splits(self, dataset, train_ratio=0.8, val_ratio=0.1):
        """Create train/validation/test splits"""
        print("Creating evaluation splits...")
        
        all_examples = []
        for split in dataset:
            all_examples.extend(list(dataset[split]))
        
        total = len(all_examples)
        train_end = int(total * train_ratio)  
        val_end = int(total * (train_ratio + val_ratio))
        
        splits = {
            "train": all_examples[:train_end],
            "validation": all_examples[train_end:val_end],
            "test": all_examples[val_end:]
        }
        
        for split_name, split_data in splits.items():
            output_file = os.path.join(self.processed_data_path, f"ds1000_{split_name}.json")
            self._save_data(split_data, output_file, "json")
            print(f"{split_name}: {len(split_data)} examples")
        
        return splits
    
    def _save_data(self, data, output_file, format_type):
        """Save data in specified format"""
        with open(output_file, 'w') as f:
            if format_type == "jsonl":
                for item in data:
                    f.write(json.dumps(item) + '\n')
            else:
                json.dump(data, f, indent=2)
    
    def get_statistics(self, dataset):
        """Get dataset statistics"""
        stats = {"total_examples": 0, "libraries": {}, "difficulties": {}}
        
        for split in dataset:
            for example in dataset[split]:
                stats["total_examples"] += 1
                
                lib = example.get("lib", "unknown")
                stats["libraries"][lib] = stats["libraries"].get(lib, 0) + 1
                
                difficulty = example.get("difficulty", "unknown") 
                stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
        
        print(f"\nDataset Statistics:")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Libraries: {dict(stats['libraries'])}")
        print(f"Difficulties: {dict(stats['difficulties'])}")
        
        return stats

class DS1000Evaluator:
    def __init__(self, processor=None):
        self.processor = processor or DS1000Processor()
    
    def evaluate_model(self, model, tokenizer, test_data, max_samples=100):
        """Evaluate model on test data"""
        if isinstance(test_data, str):
            with open(test_data, 'r') as f:
                test_data = json.load(f)
        
        test_data = test_data[:max_samples]
        results = []
        
        print(f"Evaluating on {len(test_data)} examples...")
        
        for example in tqdm(test_data, desc="Evaluating"):
            prompt = self.processor._create_prompt(example)
            generated_code = self._generate_code(model, tokenizer, prompt)
            
            result = {
                "problem_id": example.get("problem_id", ""),
                "lib": example.get("lib", ""),
                "generated_code": generated_code,
                "reference_code": example.get("reference_code", ""),
                "exact_match": generated_code.strip() == example.get("reference_code", "").strip()
            }
            results.append(result)
        
        metrics = self._calculate_metrics(results)
        self._save_results(metrics, results)
        
        print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        return metrics, results
    
    def _generate_code(self, model, tokenizer, prompt, max_new_tokens=256):
        """Generate code using the model"""
        try:
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated[len(prompt):].strip()
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return ""
    
    def _calculate_metrics(self, results):
        """Calculate evaluation metrics"""
        exact_matches = sum(1 for r in results if r["exact_match"])
        return {
            "exact_match_accuracy": exact_matches / len(results) if results else 0,
            "total_examples": len(results),
            "exact_matches": exact_matches
        }
    
    def _save_results(self, metrics, results):
        """Save evaluation results"""
        results_file = os.path.join(self.processor.processed_data_path, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump({"metrics": metrics, "results": results}, f, indent=2)

def main():
    processor = DS1000Processor()
    
    dataset = processor.download_dataset()
    processor.get_statistics(dataset)
    processor.preprocess_for_finetuning(dataset)
    splits = processor.create_splits(dataset)
    
    print("\nDataset processing complete!")
    print(f"Training examples: {len(splits['train'])}")
    print(f"Validation examples: {len(splits['validation'])}")  
    print(f"Test examples: {len(splits['test'])}")

if __name__ == "__main__":
    main() 