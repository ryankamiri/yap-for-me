"""
Configuration management for MLRL experiments.
Handles YAML loading and model configuration.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

class Config:
    """Simple configuration manager for YAML files."""

    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        self.path = Path(config_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(self.path, "r") as f:
            self.data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value using dot notation (e.g., 'model.name')."""
        keys = key.split(".")
        value = self.data

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.data.get(section, {})

    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values (deep merge)."""
        self._deep_merge(self.data, updates)

    def _deep_merge(self, base: dict, updates: dict):
        """Recursively merge updates into base dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def save(self, output_path: Optional[str] = None):
        """Save configuration to YAML file."""
        save_path = Path(output_path) if output_path else self.path
        with open(save_path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)