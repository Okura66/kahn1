# conftest.py — ensures src/ and repository root (training/, eval/) are available on sys.path
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "src"))
