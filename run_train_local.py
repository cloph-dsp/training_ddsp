"""Local runner to prepare dataset, train, and export a DDSP-VST model using a local folder ./training_data.

This script adapts the Colab notebook to run locally. It installs DDSP dependencies and runs the 
training pipeline similar to the original Train_VST.ipynb workflow.

Usage (PowerShell):
    python .\run_train_local.py --name MetalSnare --training-data ./training_data --steps 35000

Requirements:
    - Python 3.8+
    - CUDA-compatible GPU (recommended)
    - Sufficient disk space for audio processing and model checkpoints
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import os
import glob
import datetime
import warnings
import time
import config  # Load centralized configuration

# Suppress warnings that obscure output (from original notebook)
warnings.filterwarnings("ignore")


# Defaults pulled from config
DEFAULT_TRAINING_DATA = config.DEFAULT_TRAINING_DATA
DEFAULT_AUDIO_DIR = config.DEFAULT_AUDIO_DIR
DEFAULT_DATA_DIR = config.DEFAULT_DATA_DIR
DEFAULT_MODEL_BASE = config.DEFAULT_MODEL_BASE
DEFAULT_NAME = config.DEFAULT_NAME
DEFAULT_STEPS = config.DEFAULT_STEPS
IGNORE_PREVIOUS = config.IGNORE_PREVIOUS


def install_ddsp_if_needed():
    """Install DDSP with data preparation dependencies if not available."""
    try:
        import ddsp
        print("DDSP already available")
        return True
    except ImportError:
        print("Installing DDSP with data preparation dependencies...")
        print("This may take several minutes...")
        try:
            # Install ddsp with data preparation dependencies
            cmd = [sys.executable, '-m', 'pip', 'install', '-U', 'ddsp[data_preparation]']
            subprocess.check_call(cmd)
            print("DDSP installation completed successfully")
            # Add user site-packages to path so ddsp is importable in this process
            import site
            user_site = site.getusersitepackages()
            site.addsitedir(user_site)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install DDSP: {e}")
            return False


def validate_environment():
    """Validate that the environment is suitable for training."""
    print("Validating environment...")
    
    # CRITICAL: Set CUDA environment variables BEFORE TensorFlow import
    cuda_home = str(config.CUDA_HOME)
    cuda_bin = os.path.join(cuda_home, "bin")
    
    # Ensure CUDA paths are in environment
    os.environ.setdefault('CUDA_HOME', cuda_home)
    os.environ.setdefault('CUDA_PATH', cuda_home)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    
    # Add CUDA to PATH if not present
    current_path = os.environ.get('PATH', '')
    if cuda_bin not in current_path:
        os.environ['PATH'] = cuda_bin + os.pathsep + current_path
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Warning: Python 3.8+ recommended for DDSP training")
    
    # Check CUDA installation
    try:
        cuda_version = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL, text=True)
        print(f"CUDA toolkit found: {cuda_version.split('release')[1].split(',')[0].strip()}")
    except Exception:
        print("Warning: CUDA toolkit not found. Install CUDA 11.2+ for GPU acceleration.")
    
    # Check cuDNN
    try:
        # Check if cuDNN files exist (common locations)
        cudnn_paths = [
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.2\\include\\cudnn.h",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\cudnn.h",
            "/usr/include/cudnn.h"
        ]
        cudnn_found = any(os.path.exists(path) for path in cudnn_paths)
        if cudnn_found:
            print("cuDNN library found")
        else:
            print("Warning: cuDNN library not found. Install cuDNN for GPU acceleration.")
    except Exception:
        print("Warning: Could not verify cuDNN installation")
    
    # Check if GPU is available with better TensorFlow configuration
    try:
        import tensorflow as tf
        # Configure TensorFlow to be more verbose about GPU detection
        tf.config.set_soft_device_placement(True)
        tf.debugging.set_log_device_placement(True)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Try to configure memory growth
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except Exception as e:
                print(f"Warning: Could not enable memory growth: {e}")
            
            # Test GPU memory allocation
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
                    c = tf.matmul(a, b)
                    print(f"GPU test successful: {c.numpy()}")
            except Exception as e:
                print(f"Warning: GPU test failed: {e}")
                print("This might indicate CUDA/cuDNN version mismatch or driver issues")
        else:
            print("Warning: No GPU detected by TensorFlow. Training will be very slow on CPU.")
            print("Check CUDA installation and TensorFlow GPU support")
    except ImportError:
        print("Warning: TensorFlow not available. Will be installed with DDSP.")
    
    return True


def which(bin_name):
    from shutil import which as _which
    return _which(bin_name)


def check_tool(name):
    if which(name) is None:
        print(f"Warning: '{name}' not found in PATH. Please ensure ddsp is installed and the CLI is available.")
        return False
    return True


def directory_has_files(target_dir: Path):
    return any(target_dir.iterdir()) if target_dir.exists() else False


def get_audio_files(training_data_dir: Path, audio_dir: Path):
    """Copy audio files from training_data_dir to audio_dir (rename spaces to underscores).
    If training_data_dir doesn't exist or has no wav/mp3 files, raise FileNotFoundError.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    if not training_data_dir.exists():
        raise FileNotFoundError(f"Training data folder not found: {training_data_dir}")

    mp3_files = list(training_data_dir.glob("*.mp3"))
    wav_files = list(training_data_dir.glob("*.wav"))
    audio_paths = mp3_files + wav_files
    if len(audio_paths) < 1:
        raise FileNotFoundError(f"No .wav or .mp3 files found in {training_data_dir}")

    print(f"Copying {len(audio_paths)} audio files from {training_data_dir} -> {audio_dir}")
    for src in audio_paths:
        target = audio_dir / src.name.replace(' ', '_')
        shutil.copy(src, target)
        print(f"  Copied: {src} -> {target}")


def prepare_dataset_shell_call(audio_dir: Path, data_dir: Path, sample_rate=16000, frame_rate=50, example_secs=4.0, hop_secs=1.0, viterbi=True, center=True):
    """Prepare dataset using shell command approach like original notebook."""
    if directory_has_files(data_dir):
        print(f"Dataset already exists in `{data_dir}`")
        return

    print(f"Preparing new dataset from `{audio_dir}` -> `{data_dir}`")
    print()
    print('Creating dataset...')
    print('This usually takes around 2-3 minutes for each minute of audio')
    print('(10 minutes of training audio -> 20-30 minutes)')
    
    audio_filepattern = str(audio_dir / "*")
    tfrecord_path = str(data_dir / "train.tfrecord")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Use direct Python call instead of subprocess for better control
    import glob
    import logging
    try:
        from ddsp.training.data_preparation.prepare_tfrecord_lib import prepare_tfrecord
    except ImportError:
        print("Error: 'ddsp' package not found. Please install with 'pip install ddsp[data_preparation]' in your virtual environment.")
        sys.exit(1)
    
    # Enable verbose logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('tensorflow').setLevel(logging.INFO)
    
    audio_files = glob.glob(str(audio_dir / "*"))
    print(f'Found {len(audio_files)} audio files')
    
    # Set up pipeline options to avoid Windows path issues
    pipeline_opts = [
        '--temp_location=C:\\temp\\beam',
        '--staging_location=C:\\temp\\beam', 
        '--runner=DirectRunner'
    ]
    
    # Set TensorFlow logging level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    
    try:
        print("Starting dataset preparation with verbose logging...")
        prepare_tfrecord(
            input_audio_paths=audio_files,
            output_tfrecord_path=str(data_dir / "train.tfrecord"),
            num_shards=10,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            example_secs=example_secs,
            hop_secs=hop_secs,
            viterbi=viterbi,
            center=center,
            pipeline_options=pipeline_opts
        )
        print("Dataset preparation completed successfully")
    except Exception as e:
        print(f"Dataset preparation failed: {e}")
        raise


def train_shell_call(model_dir: Path, data_dir: Path, steps=30000):
    """Train model using shell command approach like original notebook."""
    # Use forward slashes for cross-platform compatibility
    file_pattern = str(data_dir / 'train.tfrecord*').replace('\\', '/')
    fp_str = f"TFRecordProvider.file_pattern='{file_pattern}'"

    # Prepare subprocess environment using centralized settings
    env = os.environ.copy()
    env.update(config.SUBPROCESS_ENV)

    # Ensure CUDA paths are set in subprocess environment
    cuda_home = str(config.CUDA_HOME)
    cuda_bin = os.path.join(cuda_home, 'bin')
    env.update({
        'CUDA_HOME': cuda_home,
        'CUDA_PATH': cuda_home,
        'CUDA_BIN_PATH': cuda_bin,
        'PATH': cuda_bin + os.pathsep + env.get('PATH', '')
    })

    # Build training command
    cmd = [
        sys.executable,
        '-m', 'ddsp.training.ddsp_run',
        '--mode=train',
        f'--save_dir={model_dir}',
        '--gin_file=models/vst/vst.gin',
        '--gin_file=datasets/tfrecord.gin',
        f'--gin_param={fp_str}',
        '--gin_param=TFRecordProvider.centered=True',
        '--gin_param=TFRecordProvider.frame_rate=50',
    f'--gin_param=batch_size={config.BATCH_SIZE}',
        f'--gin_param=train_util.train.num_steps={steps}',
        '--gin_param=train_util.train.steps_per_save=300',
        '--gin_param=trainers.Trainer.checkpoints_to_keep=3',
        '--alsologtostderr',
        '--v=1'
    ]

    # Display configuration
    print('=== Starting DDSP Training ===')
    print('Training command:', ' '.join(cmd))
    print('Environment variables for subprocess:')
    print(f"  CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    print(f"  TF_CPP_MIN_LOG_LEVEL: {env['TF_CPP_MIN_LOG_LEVEL']}")
    print(f"  TF_FORCE_GPU_ALLOW_GROWTH: {env['TF_FORCE_GPU_ALLOW_GROWTH']}")
    print('=' * 80)

    # Execute training
    try:
        subprocess.check_call(cmd, env=env)
        print('Training completed successfully')
    except subprocess.CalledProcessError as e:
        print(f'Training failed: {e}')
        raise


def reset_state(data_dir: Path, audio_dir: Path, model_dir: Path):
    # Remove and recreate data and audio directories; ensure model_dir exists
    if data_dir.exists():
        print(f"Removing existing data dir: {data_dir}")
        shutil.rmtree(data_dir)
    if audio_dir.exists():
        print(f"Removing existing audio dir: {audio_dir}")
        shutil.rmtree(audio_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)


def export_and_zip(model_dir: Path, model_name: str):
    """Export model to ONNX and create zip archive using shell command approach."""
    export_path = model_dir / model_name
    export_path.mkdir(parents=True, exist_ok=True)

    # Build export command
    cmd = [
        sys.executable,
        '-m', 'ddsp.training.ddsp_export',
        f'--name={model_name}',
        f'--model_path={model_dir}',
        f'--save_dir={export_path}',
        '--inference_model=vst_stateless_predict_controls',
        '--tflite',
        '--notfjs'
    ]
    print('Exporting model via ddsp_export...')
    print('Export command:', ' '.join(cmd))
    try:
        subprocess.check_call(cmd)
        print('Export completed successfully')
    except subprocess.CalledProcessError as e:
        print(f'Export failed: {e}')
        raise

    # Zip the exported model folder
    zip_base = str(model_dir / model_name)
    archive_path = shutil.make_archive(zip_base, 'zip', root_dir=str(model_dir), base_dir=model_name)
    print(f'Export complete. Archive created at: {archive_path}')
    return Path(archive_path)


def get_model_dir(base_dir: Path):
    base_str = 'ddsp-training'
    dirs = sorted(base_dir.glob(f"{base_str}-*"))
    if dirs and not IGNORE_PREVIOUS:
        model_dir = dirs[-1]
    else:
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
        model_dir = base_dir / f"{base_str}-{now}"
    return model_dir


def run_cli_with_python(script_name: str, args: list):
    """Run an installed console script using the current Python interpreter.

    This prefers the installed "-script.py" wrapper in the same Scripts folder as
    the console entry point so the module imports resolve using this interpreter.
    """
    from shutil import which
    cmd = None
    # Prefer running the module using the same Python interpreter to ensure imports
    module_map = {
        'ddsp_prepare_tfrecord': 'ddsp.training.data_preparation.ddsp_prepare_tfrecord',
        'ddsp_run': 'ddsp.training.ddsp_run',
        'ddsp_export': 'ddsp.training.ddsp_export',
    }
    module = module_map.get(script_name)
    if module:
        cmd = [sys.executable, '-m', module] + args
    else:
        # Fallback to locating an installed console script/exe
        script_exe = which(script_name)
        if script_exe:
            # Try to find the -script.py wrapper next to the exe (Windows installs both)
            script_py = None
            if script_exe.lower().endswith('.exe'):
                candidate = script_exe[:-4] + '-script.py'
                if os.path.exists(candidate):
                    script_py = candidate
            # If we found a python wrapper script, run it with current interpreter
            if script_py:
                cmd = [sys.executable, script_py] + args
            else:
                # Fallback: run the exe directly
                cmd = [script_exe] + args
        else:
            raise FileNotFoundError(f"Could not locate CLI for {script_name}")

    print('Running command:', ' '.join(cmd))
    subprocess.check_call(cmd)


def get_gpu_type():
    try:
        # Check nvidia-smi
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"], 
                                       stderr=subprocess.DEVNULL, text=True)
        lines = [l for l in output.splitlines() if l.strip()]
        if len(lines) > 1:
            gpu_info = lines[1]
            print(f"NVIDIA GPU detected: {gpu_info}")
            return gpu_info
    except Exception as e:
        print(f"No NVIDIA GPU detected via nvidia-smi: {e}")
    
    # Check TensorFlow GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"TensorFlow detected {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu}")
            return f"TensorFlow GPU: {len(gpus)} device(s)"
        else:
            print("TensorFlow: No GPU devices detected")
    except ImportError:
        print("TensorFlow not available for GPU detection")
    except Exception as e:
        print(f"TensorFlow GPU detection failed: {e}")
    
    return 'CPU only'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=DEFAULT_NAME, help='Model name')
    parser.add_argument('--training-data', default=str(DEFAULT_TRAINING_DATA), help='Folder with training audio (.wav/.mp3)')
    parser.add_argument('--steps', type=int, default=DEFAULT_STEPS, help='Training steps')
    parser.add_argument('--base-dir', default=str(DEFAULT_MODEL_BASE), help='Base dir for model output (default .)')
    args = parser.parse_args()

    # Ensure DDSP package is installed in this environment
    try:
        import ddsp
    except ImportError:
        print("'ddsp' package not found. Installing now into current environment...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', 'ddsp[data_preparation]'])
            # Ensure newly installed package is importable
            import site
            site.addsitedir(site.getusersitepackages())
            import ddsp
            print("Successfully installed ddsp")
        except Exception as e:
            print(f"Failed to install ddsp: {e}")
            sys.exit(1)
    
    # Enable verbose logging globally
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # CRITICAL: Set CUDA environment variables BEFORE any TensorFlow imports
    # These must be set at the very beginning before TensorFlow is loaded
    
    # CUDA paths (already in PATH, but let's ensure they're set)
    cuda_home = str(config.CUDA_HOME)
    cuda_bin = os.path.join(cuda_home, "bin")
    cuda_lib = os.path.join(cuda_home, "lib", "x64")
    cuda_include = os.path.join(cuda_home, "include")
    
    # Add CUDA paths to PATH if not already there
    current_path = os.environ.get('PATH', '')
    if cuda_bin not in current_path:
        os.environ['PATH'] = cuda_bin + os.pathsep + current_path
    
    # Set CUDA environment variables
    os.environ['CUDA_HOME'] = cuda_home
    os.environ['CUDA_PATH'] = cuda_home
    os.environ['CUDA_BIN_PATH'] = cuda_bin
    os.environ['CUDA_LIB_PATH'] = cuda_lib
    os.environ['CUDA_INCLUDE_PATH'] = cuda_include
    
    # TensorFlow GPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable all TensorFlow logging
    os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_VMODULE'] = 'gpu_device=1,gpu_device_placer=1'
    
    # Ensure CUDA is available and visible
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
    
    # Additional CUDA environment variables for cuDNN
    os.environ['CUDNN_PATH'] = cuda_home
    os.environ['CUDNN_BIN_PATH'] = cuda_bin
    os.environ['CUDNN_LIB_PATH'] = cuda_lib
    
    print("=== CUDA Environment Setup ===")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"CUDA in PATH: {'Yes' if cuda_bin in os.environ.get('PATH', '') else 'No'}")
    print("=" * 50)
    
    # Validate environment before proceeding
    validate_environment()
    print()
    
    name = args.name
    training_data_dir = Path(args.training_data)
    audio_dir = DEFAULT_AUDIO_DIR
    data_dir = DEFAULT_DATA_DIR
    base_dir = Path(args.base_dir)

    gpu = get_gpu_type()
    print(f"GPU: {gpu or 'None detected (CPU)'}\n")

    # Check ddsp CLI tools
    check_tool('ddsp_prepare_tfrecord')
    check_tool('ddsp_run')
    check_tool('ddsp_export')

    model_dir = get_model_dir(base_dir)
    print(f"Using model dir: {model_dir}")

    reset_state(data_dir, audio_dir, model_dir)
    get_audio_files(training_data_dir, audio_dir)
    
    # Prepare dataset using shell command approach
    prepare_dataset_shell_call(audio_dir, data_dir)
    
    # Train model using shell command approach  
    train_shell_call(model_dir, data_dir, steps=args.steps)
    
    # Export and zip model
    archive = export_and_zip(model_dir, name)
    print(f"Done. Model archive at: {archive}")


if __name__ == '__main__':
    main()
