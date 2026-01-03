# Installation (macOS)

## 1) Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2.1) Audio libs (for sounddevice/soundfile)

```bash
brew install portaudio libsndfile
```

## 3) Ensure Python 3.11

```bash
uv python install 3.11
```

## 4) Create and activate venv with uv

```bash
uv venv .venv
source .venv/bin/activate
```

## 5) Install dependencies

```bash
uv pip install -r requirements.txt
```

## 6) Run the app

```bash
python launcher.py
```

## 8) macOS mic permission

- Allow microphone in System Settings → Privacy & Security → Microphone on first prompt.
