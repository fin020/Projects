from .json_loader import load_tax_config


config = load_tax_config("uk_tax/data/2025_2026.json")
version = "1.0.0"
print(f"Welcome to my tax calculator {version}")