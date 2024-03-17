# LINC: A Reimplementation

Reimplementation of [LINC Paper](https://aclanthology.org/2023.emnlp-main.313.pdf) for CS5134.

## Setup

Make sure you have poetry installed.

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