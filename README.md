# DDSP-VST Local Training Pipeline


## Quick Setup (Recommended)

1. **Install Python 3.10** (from https://www.python.org/downloads/)
2. **Create a virtual environment in VS Code:**
   - Open Command Palette (`Ctrl+Shift+P`)
   - Run `Python: Create Environment` and select Python 3.10
3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Training Script

To train a model using your audio files in `training_data/`:
```powershell
python run_train_local.py --name MetalSnare --training-data "d:\Github\training_ddsp\training_data" --steps 35000
```
- You can change `--name` and `--steps` as needed.

## Configuration

All default paths and environment variables are centralized in `config.py`. You can customize:

- **Paths and names**: `DEFAULT_TRAINING_DATA`, `DEFAULT_AUDIO_DIR`, `DEFAULT_DATA_DIR`, `DEFAULT_MODEL_BASE`, `DEFAULT_NAME`, `DEFAULT_STEPS`
- **Subprocess environment**: the `SUBPROCESS_ENV` dict holds variables like `TF_CPP_MIN_LOG_LEVEL`, `CUDA_VISIBLE_DEVICES`, `TF_FORCE_GPU_ALLOW_GROWTH`
- **CUDA path**: set `CUDA_HOME` to your local CUDA installation directory

Edit `config.py` to adjust these values before running the training script.

## Notes
- The folders `audio/`, `data/`, and `ddsp-training-*` are generated and can be safely deleted between runs.
- For advanced usage, see comments in `run_train_local.py`.

## Troubleshooting
- If you see dependency errors, make sure you are using Python 3.10 and the provided requirements.txt.
- For advanced usage, see comments in `run_train_local.py`.

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'ddsp'`, make sure you installed DDSP inside the activated venv.
- If you see errors about `crepe`, install version `0.0.16` before installing DDSP.
- For CUDA/GPU support, ensure your drivers and TensorFlow are properly configured.

# Run DDSP-VST training locally

This repository contains a Colab notebook-derived script. To run training locally, use the included `run_train_local.py` script which expects a local folder `./training_data` containing `.wav` or `.mp3` files.

Prerequisites:
- Python 3.8+
- The DDSP package installed so the CLI commands `ddsp_prepare_tfrecord`, `ddsp_run`, and `ddsp_export` are available in your PATH (e.g. `pip install -U ddsp[data_preparation]`).
- (Optional) NVIDIA drivers and CUDA if you want GPU acceleration.

Quick start (PowerShell):

```powershell
# From repo root
python .\run_train_local.py --name MetalSnare --training-data .\training_data --steps 35000
```

If `ddsp_*` commands are not found, install ddsp or run inside a Python environment where ddsp is installed.

The script will:
- Copy audio from `./training_data` -> `./audio`
- Prepare TFRecords in `./data`
- Train saving checkpoints to `./ddsp-training-<timestamp>`
- Export and create a zip archive of the exported model
