import abc
import time
import torch
import requests
import json
from ollama import chat
from ollama import ChatResponse
from decouple import config
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

class Llama3:
    """Class for interacting with the Ollama API."""
    def __init__(self):
        self.base_url = "http://localhost:11434"  # Ollama default host
        self.model_name = "llama3:8b"

    def submit_request(self, prompt, temperature=0.6, max_tokens=1024, n=1, split_by=None):
        """Submit a request to the Ollama API."""
        error_counter = 0
        
        while True:
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data)
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed. Status code: {response.status_code}")
                
                response_data = response.json()
                generated_text = response_data.get("response", "").strip()
                
                return [generated_text] * n
                
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(1)
                error_counter += 1
                if error_counter > 10:
                    raise e
                if 'filtered' in str(e).lower():
                    return [''] * n


class Qwen2:
    def __init__(self):
        self.base_url = "http://localhost:11434"  # Ollama default host
        self.model_name = "qwen2:7b"

    def submit_request(self, prompt, temperature=0.6, max_tokens=1024, n=1, split_by=None):
        error_counter = 0
        
        while True:
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data)
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed. Status code: {response.status_code}")
                
                response_data = response.json()
                generated_text = response_data.get("response", "").strip()
                
                return [generated_text] * n
                
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(1)
                error_counter += 1
                if error_counter > 10:
                    raise e
                if 'filtered' in str(e).lower():
                    return [''] * n


class Phi3:
    def __init__(self):
        self.base_url = "http://localhost:11434"  # Ollama default host
        self.model_name = "phi3:8b"

    def submit_request(self, prompt, temperature=0.6, max_tokens=1024, n=1, split_by=None):
        error_counter = 0
        
        while True:
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data)
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed. Status code: {response.status_code}")
                
                response_data = response.json()
                generated_text = response_data.get("response", "").strip()
                
                return [generated_text] * n
                
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(1)
                error_counter += 1
                if error_counter > 10:
                    raise e
                if 'filtered' in str(e).lower():
                    return [''] * n

class Gemma2:
    def __init__(self):
        self.base_url = "http://localhost:11434"  # Ollama default host
        self.model_name = "gemma2:9b"

    def submit_request(self, prompt, temperature=0.6, max_tokens=1024, n=1, split_by=None):
        error_counter = 0
        
        while True:
            try:
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    },
                    "stream": False
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data)
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed. Status code: {response.status_code}")
                
                response_data = response.json()
                generated_text = response_data.get("response", "").strip()
                
                return [generated_text] * n
                
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(1)
                error_counter += 1
                if error_counter > 10:
                    raise e
                if 'filtered' in str(e).lower():
                    return [''] * n


class EmbeddingModel:
    @abc.abstractmethod
    def submit_embedding_request(self, text):
        """
        submit embedding request by the given model
        @param text: text to be embedded
        @return: embedding vector of the text
        """
        pass


class GPTEmbedding(EmbeddingModel):
    """Class for interacting with the Ollama API Embedding model."""
    def __init__(self):
        self.base_url = "http://localhost:11434"  # Ollama default host
        self.model_name = "text-embedding-ada-002"

    def submit_embedding_request(self, text):
        """Submit a request to the Ollama API."""
        error_counter = 0
        
        while True:
            try:
                data = {
                    "model": self.model_name,
                    "prompt": text, 
                    "options": {
                        "temperature": 0  
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/api/embeddings", 
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(data)
                )
                
                if response.status_code != 200:
                    raise Exception(f"API request failed. Status code: {response.status_code}")
                
                # 解析响应
                response_data = response.json()
                embedding = response_data.get("embedding", [])
                
                if not embedding:
                    raise Exception("No valid embedding vector was obtained")
                
                return embedding
                
            except Exception as e:
                print(f"Request error: {str(e)}")
                time.sleep(1)
                error_counter += 1
                if error_counter > 10:
                    raise e


class SBert(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)

        return response


class E5(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('intfloat/e5-large-v2')

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)

        return response


class BERTembedding(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer("bert-base-uncased")

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)

        return response
