# LINC: A Reimplementation

Reimplementation of [LINC Paper](https://aclanthology.org/2023.emnlp-main.313.pdf) for CS5134.

# Setup

Make sure you have poetry installed.

## Installing Packages

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

## API Keys

To use the API Models in `models.py`, we need some API keys setup in the local env. Adhering to the naming conventions in `.env.example`, create a `.env` file in the root folder and add in the corresponding API key.

### Google API Key

Go to [this site](https://aistudio.google.com/app/apikey) to get a Gemini API key.

```
GOOGLE_API_KEY="API KEY HERE"
```

### Cohere API Key

Go to [this site](https://dashboard.cohere.com/api-keys) to get a Cohere API key.

```
COHERE_API_KEY="API KEY HERE"
```