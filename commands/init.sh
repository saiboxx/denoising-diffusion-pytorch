#! /bin/bash

python -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install einops

python -u main.py
