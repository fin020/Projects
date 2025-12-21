import os
from .json_loader import load_tax_config



current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "data", "2025_2026.json")


config = load_tax_config("uk_tax/data/2025_2026.json")