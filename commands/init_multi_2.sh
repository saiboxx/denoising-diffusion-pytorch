#! /bin/bash

python -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install einops
python -m pip install pyyaml

python -u main.py -c configs/02_config.yml -s $1