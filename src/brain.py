import os
import yaml
from typing import Dict, List, Generator
import ollama

class AIBrain:
    """Manages the connection to Ollama, model checks, and text generation/streaming."""
    
    def __init__(self, config_path: str = "config/personality.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.model_name = self.config.get("model_name", "llama3.2:latest")
        self.embedding_model = self.config.get("embedding_model_name", "nomic-embed-text")
        self.vision_model_name = self.config.get("vision_model_name", "llava:latest")
        self.system_prompt = self.config.get("system_prompt", "You are Kiri, a helpful assistant.")
        self.name = self.config.get("name", "Kiri")
        
        # Verify connection and ensure models are available
        self._ensure_models_available()

    def _load_config(self) -> Dict:
        """Loads the YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _ensure_models_available(self):
        """Checks if the required LLM and embedding models are loaded locally in Ollama."""
        print(f"[{self.name} Brain] Connecting to local Ollama service...")
        try:
            res = ollama.list()
            downloaded_models = []
            
            # Support both dict and ListResponse objects from different ollama library versions
            if isinstance(res, dict):
                models_list = res.get("models", [])
            else:
                models_list = getattr(res, "models", [])
                
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get("name", m.get("model", ""))
                else:
                    name = getattr(m, "model", getattr(m, "name", ""))
                if name:
                    downloaded_models.append(name)
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama: {e}. Please make sure the Ollama application is running locally."
            ) from e


        for model in [self.model_name, self.embedding_model]:
            if model not in downloaded_models and f"{model}:latest" not in downloaded_models:
                print(f"[{self.name} Brain] Model '{model}' not found locally. Pulling from Ollama registry...")
                try:
                    current_status = ""
                    for progress in ollama.pull(model, stream=True):
                        status = progress.get("status", "")
                        if status != current_status:
                            print(f"[{self.name} Brain] Pulling '{model}': {status}")
                            current_status = status
                    print(f"[{self.name} Brain] Successfully pulled '{model}'!")
                except Exception as e:
                    print(f"[{self.name} Brain] Error pulling '{model}': {e}")
                    print(f"[{self.name} Brain] Please verify model name or try running 'ollama pull {model}' in terminal.")

        # Check for optional vision model and print a notification if it's missing (no auto-pull)
        v_model = self.vision_model_name
        if v_model not in downloaded_models and f"{v_model}:latest" not in downloaded_models:
            print(f"[{self.name} Brain] Note: Optional vision model '{v_model}' not found locally.")
            print(f"[{self.name} Brain] Commands like '/see' will dynamically download it on first use.")

    def generate_response(self, conversation_history: List[Dict[str, str]], stream: bool = True) -> Generator[str, None, None] | str:
        """Generates response based on conversation history (including system prompt)."""
        # Inject system prompt at the very beginning of conversation history
        messages = [{"role": "system", "content": self.system_prompt}] + conversation_history
        
        if stream:
            return self._generate_response_stream(messages)
        else:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                stream=False
            )
            return response["message"]["content"]

    def _generate_response_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in response:
            yield chunk["message"]["content"]

    def ensure_vision_model_available(self):
        """Ensures the multimodal vision model is downloaded, pulling it if necessary."""
        try:
            res = ollama.list()
            downloaded = []
            if isinstance(res, dict):
                models_list = res.get("models", [])
            else:
                models_list = getattr(res, "models", [])
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get("name", m.get("model", ""))
                else:
                    name = getattr(m, "model", getattr(m, "name", ""))
                if name:
                    downloaded.append(name)
            
            model = self.vision_model_name
            if model not in downloaded and f"{model}:latest" not in downloaded:
                print(f"\n[{self.name} Brain] Vision model '{model}' is missing.")
                print(f"[{self.name} Brain] Downloading '{model}' from Ollama registry (this may take a few minutes)...")
                current_status = ""
                for progress in ollama.pull(model, stream=True):
                    status = progress.get("status", "")
                    if status != current_status:
                        print(f"[{self.name} Brain] Downloading '{model}': {status}")
                        current_status = status
                print(f"[{self.name} Brain] Vision model '{model}' successfully downloaded!")
        except Exception as e:
            print(f"[{self.name} Brain] Error checking/downloading vision model: {e}")

    def generate_vision_response(self, prompt: str, image_path: str, stream: bool = True) -> Generator[str, None, None] | str:
        """Generates a response for a visual query with an image file."""
        messages = [{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }]
        
        if stream:
            return self._generate_vision_stream(messages)
        else:
            response = ollama.chat(
                model=self.vision_model_name,
                messages=messages,
                stream=False
            )
            return response["message"]["content"]

    def _generate_vision_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        response = ollama.chat(
            model=self.vision_model_name,
            messages=messages,
            stream=True
        )
        for chunk in response:
            yield chunk["message"]["content"]
