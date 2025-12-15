"""
Pytest configuration for OpenRouter client tests.

This conftest.py registers custom models from prod_env/model_metadata.yaml
with HELM's model metadata registry before tests run.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import prod_env
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import HELM's registry
from helm.benchmark.model_metadata_registry import register_model_metadata_from_path

# Register custom models from prod_env/model_metadata.yaml
model_metadata_path = project_root / "prod_env" / "model_metadata.yaml"
if model_metadata_path.exists():
    register_model_metadata_from_path(str(model_metadata_path))
    print(f"✓ Registered custom models from {model_metadata_path}")
else:
    print(f"⚠ Warning: {model_metadata_path} not found")
