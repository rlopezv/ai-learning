import requests
import os

class Generator:
    def __init__(self, model: str = "mistral"):
        self.model = model
        base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.url = f"{base_url}/api/generate"

    def generate(self, prompt: str) -> str:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]
