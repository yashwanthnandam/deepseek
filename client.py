"""
Simple client to test the DeepSeek server
"""

import requests
import json
import asyncio
import aiohttp

class DeepSeekClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check if the server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """Generate text using the simple endpoint"""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                params={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def chat_completion(self, messages: list, max_tokens: int = 512, temperature: float = 0.7):
        """Chat completion using OpenAI-compatible endpoint"""
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    client = DeepSeekClient()
    
    # Health check
    print("=== Health Check ===")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if health.get("status") != "healthy":
        print("Server is not healthy. Please check the logs.")
        return
    
    # Simple text generation
    print("\n=== Simple Text Generation ===")
    prompt = "def fibonacci(n):"
    result = client.generate_text(prompt, max_tokens=256, temperature=0.3)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result.get('generated_text', 'Error')}")
    
    # Chat completion
    print("\n=== Chat Completion ===")
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function to calculate the factorial of a number."}
    ]
    
    chat_result = client.chat_completion(messages, max_tokens=256, temperature=0.3)
    if "choices" in chat_result:
        response_content = chat_result["choices"][0]["message"]["content"]
        print(f"Assistant: {response_content}")
    else:
        print(f"Error: {chat_result}")

if __name__ == "__main__":
    main()