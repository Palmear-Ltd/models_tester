#!/bin/bash
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Error: Virtual environment not found!"
  echo "Please run: uv venv .venv && source .venv/bin/activate && uv pip install -r requirements.txt"
  read -p "Press Enter to exit..."
  exit 1
fi

# Launch app in foreground (UI shows) and log output; close terminal when app exits.
exec ./.venv/bin/python launcher.py >./palmear_app.log 2>&1

