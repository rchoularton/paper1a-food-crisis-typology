#!/bin/bash
set -e

# Code Ocean entry point
# On Code Ocean: data is at /data, results go to /results
# Our scripts expect data/ and outputs/ relative to run_all.py
if [ -d "../results" ]; then
    ln -sf ../results outputs
fi
if [ -d "../data" ] && [ ! -d "data" ]; then
    ln -sf ../data data
fi

python -u run_all.py
