import httpx
from typing import Dict, Any, Optional
from .config import BackendConfig


class ModelClient:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.base_url = config.gpu_server_url.rstrip("/")
        self.api_key = config.gpu_server_api_key
        self.server_type = config.gpu_server_type.lower()
        
        inference_config = config.get_inference_config()
        if "model_request_timeout" not in inference_config:
            raise ValueError("model_request_timeout is required in inference config")
        self.timeout = httpx.Timeout(inference_config["model_request_timeout"])
        
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def _request_vllm(
        self,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> str:
        url = f"{self.base_url}/v1/completions"
        
        required_keys = ["max_tokens", "temperature", "top_p", "top_k"]
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"{key} is required in model config")
        
        payload = {
            "prompt": prompt,
            "temperature": model_config["temperature"],
            "max_tokens": model_config["max_tokens"],
            "top_p": model_config["top_p"],
            "top_k": model_config["top_k"]
        }
        
        if "repetition_penalty" in model_config:
            payload["repetition_penalty"] = model_config["repetition_penalty"]
        
        if "stop_sequences" in model_config:
            payload["stop"] = model_config["stop_sequences"]
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"].strip()
            return ""
    
    async def _request_ollama(
        self,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> str:
        url = f"{self.base_url}/api/generate"
        
        required_keys = ["max_tokens", "temperature", "top_p", "top_k"]
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"{key} is required in model config")
        
        options = {
            "temperature": model_config["temperature"],
            "num_predict": model_config["max_tokens"],
            "top_p": model_config["top_p"],
            "top_k": model_config["top_k"]
        }
        
        if "repetition_penalty" in model_config:
            options["repeat_penalty"] = model_config["repetition_penalty"]
        
        payload = {
            "prompt": prompt,
            "options": options
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            if "response" not in result:
                raise ValueError("Ollama API response missing 'response' field")
            return result["response"].strip()
    
    async def _request_tgi(
        self,
        prompt: str,
        model_config: Dict[str, Any]
    ) -> str:
        url = f"{self.base_url}/generate"
        
        required_keys = ["max_tokens", "temperature", "top_p", "top_k"]
        for key in required_keys:
            if key not in model_config:
                raise ValueError(f"{key} is required in model config")
        
        parameters = {
            "temperature": model_config["temperature"],
            "max_new_tokens": model_config["max_tokens"],
            "top_p": model_config["top_p"],
            "top_k": model_config["top_k"]
        }
        
        if "repetition_penalty" in model_config:
            parameters["repetition_penalty"] = model_config["repetition_penalty"]
        
        if "stop_sequences" in model_config:
            parameters["stop"] = model_config["stop_sequences"]
        
        payload = {
            "inputs": prompt,
            "parameters": parameters
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "generated_text" in result:
                generated = result["generated_text"]
                if generated.startswith(prompt):
                    return generated[len(prompt):].strip()
                return generated.strip()
            return ""
    
    async def infer(
        self,
        context: str,
        new_message: str,
        response_prefix: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        if config is None:
            config = self.config.get_model_config()
        
        full_prompt = f"{context}\n{new_message}\n{response_prefix}"
        
        try:
            if self.server_type == "vllm":
                return await self._request_vllm(full_prompt, config)
            elif self.server_type == "ollama":
                return await self._request_ollama(full_prompt, config)
            elif self.server_type == "tgi":
                return await self._request_tgi(full_prompt, config)
            else:
                raise ValueError(
                    f"Unsupported GPU server type: {self.server_type}. "
                    "Supported types: vllm, ollama, tgi"
                )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Model server error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to model server: {str(e)}")

