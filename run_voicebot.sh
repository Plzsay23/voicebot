#!/usr/bin/env bash
set -e

cd ~/voicebot
source .venv/bin/activate

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

python chatbot.py
