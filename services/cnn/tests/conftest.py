"""Global pytest configuration."""
import sys
from pathlib import Path

# Ensure `tests.fixtures.*` is importable
sys.path.insert(0, str(Path(__file__).parent))
