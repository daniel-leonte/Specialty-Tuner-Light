"""
Model loading and setup for code generation models.

This module provides unified interfaces for loading DeepSeek Coder models
with support for both PyTorch and MLX backends. Features include:
- Automatic device detection and optimization
- PyTorch implementation for CUDA/CPU
- MLX implementation for Apple Silicon
- Factory function for automatic backend selection
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekCoder:
    """PyTorch-based DeepSeek Coder implementation"""
    
    def __init__(self, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    
    def generate(self, prompt, max_length=512, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_length=max_length, temperature=temperature, 
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class DeepSeekCoderMLX:
    """MLX-based DeepSeek Coder implementation for Apple Silicon"""
    
    def __init__(self, model_name="mlx-community/deepseek-coder-1.3b-instruct-mlx"):
        try:
            from mlx_lm import load, generate as mlx_generate
            self.model, self.tokenizer = load(model_name)
            self.mlx_generate = mlx_generate
        except ImportError:
            raise ImportError("MLX not available. Install with: pip install mlx-lm")
    
    def generate(self, prompt):
        return self.mlx_generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

def get_model(backend="auto", model_name=None):
    """Factory function to get the appropriate model based on backend preference"""
    if backend == "auto":
        # Auto-select based on platform
        if torch.backends.mps.is_available():
            try:
                return DeepSeekCoderMLX(model_name) if model_name else DeepSeekCoderMLX()
            except ImportError:
                return DeepSeekCoder(model_name) if model_name else DeepSeekCoder()
        else:
            return DeepSeekCoder(model_name) if model_name else DeepSeekCoder()
    elif backend == "pytorch":
        return DeepSeekCoder(model_name) if model_name else DeepSeekCoder()
    elif backend == "mlx":
        return DeepSeekCoderMLX(model_name) if model_name else DeepSeekCoderMLX()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'pytorch', or 'mlx'")

if __name__ == "__main__":
    coder = get_model()
    response = coder.generate("Write a Python function to calculate fibonacci numbers:")
    print(response) 