# DDSP Training that works


## Prerequisites

1. Macbook with Apple Silicon
2. Python 3.8 (using `uv` package manager).
   1. Create a virtual evironment before running this notebook by `uv venv --python 3.8`, which creates a `.venv` directory. Activate the venv by `source .venv/bin/activate`.
   2. Install jupyter notebook by `uv pip install jupyter notebook`.  
3. Local-install `ddsp`
   1. `git clone https://github.com/magenta/ddsp.git`, make sure the commit is `a2d0517`.
   2. Modify `install_requires` in `setup.py` by removing all version restriction.
   3. Install locally by `uv pip install -e ./ddsp`
4. Install `tensorflow-metal` by `uv pip install tensorflow-metal` to allow GPU usage in tensorflow.
