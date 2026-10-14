import sys
from pathlib import Path

# Make `src` and `app` importable when pytest runs from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
