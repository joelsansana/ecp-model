import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:  # allow `pytest` without an editable install
    sys.path.insert(0, str(SRC))
