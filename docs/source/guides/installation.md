# Installation Guide

## Prerequisites

- Python (recommended: 3.10+)
- PyTorch
- NumPy / SciPy
- Matplotlib

Notes:

- Some project dependencies are scientific packages that may install faster via Conda.
- The repository root `requirements.txt` uses a Conda-style format (e.g. `python=3.12`). If you use `pip`, prefer installing the package itself (`pip install -e .`) and let `pyproject.toml` / your environment resolve dependencies.

## Install from PyPI

The easiest way to install AI4Plasma is via pip:

```bash
pip install ai4plasma
```

Or upgrade to the latest version:

```bash
pip install --upgrade ai4plasma
```

## Install from Source

```bash
git clone https://github.com/ai4plasma/ai4plasma.git
cd ai4plasma
pip install -e .
```

## Install by Conda

If you use Conda/Mamba:

```bash
conda create -n ai4plasma python=3.12
conda activate ai4plasma
pip install -e .
```

## Build documentation locally

The documentation is built with Sphinx + MyST.

```bash
pip install -r docs/requirements.txt
cd docs
make html
```

On Windows you can also use:

```bat
cd docs
make.bat html
```
