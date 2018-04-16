#!/bin/bash

# Load anaconda environment
module load anaconda

# activate conda netflix
source activate netflix

# run linear regressions on the data
cd /home/thomasrr/netflix-ml
echo linear_regression.py --movie-offset $1 --num-movies 1770 --choosing-mechanism random_sample --max-reviews 1000 --num-cores 10

