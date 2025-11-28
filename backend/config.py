import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv

load_dotenv()


class Config:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "inference_config.yaml"
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, "r") as f:
            self.data = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        return self.data.get(section, {})


class BackendConfig:
    def __init__(self):
        self.bluebubbles_url = self._get_required_env("BLUEBUBBLES_URL")
        self.bluebubbles_password = self._get_required_env("BLUEBUBBLES_PASSWORD")
        
        self.gpu_server_url = self._get_required_env("GPU_SERVER_URL")
        self.gpu_server_api_key = self._get_required_env("GPU_SERVER_API_KEY")
        self.gpu_server_type = self._get_required_env("GPU_SERVER_TYPE")
        
        self.backend_host = self._get_required_env("BACKEND_HOST")
        self.backend_port = int(self._get_required_env("BACKEND_PORT"))
        
        self.inference_config = Config()
    
    def _get_required_env(self, key: str) -> str:
        value = os.getenv(key)
        if value is None or value == "":
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.inference_config.get_section("model")
    
    def get_inference_config(self) -> Dict[str, Any]:
        return self.inference_config.get_section("inference")

