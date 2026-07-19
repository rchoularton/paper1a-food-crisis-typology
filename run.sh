#!/bin/bash
set -e

# Code Ocean entry point
# On Code Ocean: data is at /data, results go to /results
# Our scripts expect data/ and outputs/ relative to run_all.py
if [ -d "../results" ]; then
    # `ln -sf ../results outputs` only redirects when `outputs` does not already
    # exist as a real directory. If it does, ln creates `outputs/results` instead
    # and every artifact is written to /code/outputs, which Code Ocean discards
    # at the end of a run — leaving /results with nothing but the console log.
    # Remove any real outputs/ first so artifacts land where they are captured.
    if [ -d "outputs" ] && [ ! -L "outputs" ]; then
        echo "run.sh: removing pre-existing outputs/ so /results capture works"
        rm -rf outputs
    fi
    ln -sfn ../results outputs
fi
if [ -d "../data" ] && [ ! -d "data" ]; then
    ln -sf ../data data
fi

python -u run_all.py
