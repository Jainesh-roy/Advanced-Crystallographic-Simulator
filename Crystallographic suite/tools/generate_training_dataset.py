from pathlib import Path
import sys
from core.ml.dataset_generator import save_dataset
from core.ml.config import SAMPLES_PER_MATERIAL

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.ml.dataset_generator import test

if __name__ == "__main__":
    save_dataset(samples_per_material=SAMPLES_PER_MATERIAL)