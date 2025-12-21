from json_loader import load_tax_config
from pathlib import Path

# Get the parent directory of modules/
package_dir = Path(__file__).parent.parent
config_path = package_dir / "data" / "2025_2026.json"
config = load_tax_config(str(config_path))