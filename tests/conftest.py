import sys
from pathlib import Path
import nltk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
nltk.data.path.insert(0, str(ROOT / ".cache/nltk_data"))
