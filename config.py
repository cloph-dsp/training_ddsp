from pathlib import Path

# Default directories and names
DEFAULT_TRAINING_DATA = Path("training_data")
DEFAULT_AUDIO_DIR = Path("audio")
DEFAULT_DATA_DIR = Path("data")
DEFAULT_MODEL_BASE = Path(".")
DEFAULT_NAME = "MetalSnare"
DEFAULT_STEPS = 35000
IGNORE_PREVIOUS = False

# Environment variables for subprocess training
SUBPROCESS_ENV = {
    'TF_CPP_MIN_LOG_LEVEL': '0',
    'CUDA_VISIBLE_DEVICES': '0',
    'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
}

# CUDA installation path
CUDA_HOME = Path(r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2")
# Default gin batch size for training
BATCH_SIZE = 2
