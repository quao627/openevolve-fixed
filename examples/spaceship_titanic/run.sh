#!/bin/bash
# Run OpenEvolve on the Spaceship Titanic classification task
cd "$(dirname "$0")/../.."
python openevolve-run.py \
    examples/spaceship_titanic/initial_program.py \
    examples/spaceship_titanic/evaluator.py \
    --config examples/spaceship_titanic/config.yaml \
    --iterations 200
