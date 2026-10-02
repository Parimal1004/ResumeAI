"""
pytest configuration — adds the project root to sys.path so that
`from backend.X import Y` works from within the tests/ directory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
