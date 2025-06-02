import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DeepSeekCoder:
    def __init__(self, model_name="deepseek-ai/deepseek-coder-6.7b-instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

# Usage example
if __name__ == "__main__":
    coder = DeepSeekCoder()
    response = coder.generate("Write a Python function to calculate fibonacci numbers:")
    print(response) 