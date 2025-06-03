from mlx_lm import load, generate

class DeepSeekCoderMLX:
    def __init__(self, model_name="mlx-community/deepseek-coder-1.3b-instruct-mlx"):
        self.model, self.tokenizer = load(model_name)
    
    def generate(self, prompt):
        return generate(self.model, self.tokenizer, prompt=prompt, verbose=True)

# Usage example
if __name__ == "__main__":
    coder = DeepSeekCoderMLX()
    response = coder.generate("Write a Python function to calculate fibonacci numbers:")
    print(response) 