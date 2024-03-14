# LINC: A Reimplementation

Reimplementation of [LINC Paper](https://aclanthology.org/2023.emnlp-main.313.pdf) for CS5134.

## Setup

Make sure you have poetry installed.

### Dependency

This project assumes Windows and CUDA 12.1+

For Linux and the same CUDA version, replace the `torch` line in `pyproject.toml` with 

```toml
torch = { url = "https://download.pytorch.org/whl/cu121/fbgemm_gpu-0.6.0%2Bcu121-cp310-cp310-manylinux2014_x86_64.whl" }
```

This assumes your cuda version is 12.1+. For other CUDA versions and OS combinations, check [this post](https://github.com/python-poetry/poetry/issues/6409) for installation instructions.

This is done this way because alternative methods of installing torch will end up with poetry installing every single possible `torch` wheels, this is **intended behavior**. Since we don't want to sit around for that, this workaround is faster.

### Installing Packages

Installing for usage:

```terminal
poetry install
```

Installing for development:

```terminal
poetry install --with dev
poetry run pre-commit install
```

## Downloading Dataset

For Linux, run

```bash
bash install_dataset.sh
```

For Windows, run

```powershell
.\install_dataset.ps1
```