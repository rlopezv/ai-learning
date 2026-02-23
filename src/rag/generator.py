from transformers import pipeline

class Generator:
    def __init__(self):
        self.generator = pipeline("text-generation", model="distilgpt2")

    def generate(self, prompt: str) -> str:
        result = self.generator(prompt, max_length=200, num_return_sequences=1)
        return result[0]["generated_text"]
