#!/bin/bash
# parallel job using 10 processors. and runs for 4 hours (max)
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=10
#SBATCH -t 2:00:00
#SBATCH --array=0-9
#SBATCH --job-name=linear_regression
#SBATCH --mem 50000
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=thomasrr@princeton.edu

# Load anaconda environment
module load anaconda

# activate conda netflix
source activate netflix

# run linear regressions on the data
cd /home/thomasrr/netflix-ml
python linear_regression.py --movie-offset $((1 + SLURM_ARRAY_TASK_ID*1777)) --num-movies 1777 --regr-type Ridge --choosing-mechanism random_sample --max-reviews 1000 --num-cores 10 --ofile pickle/ridge_results_$((SLURM_ARRAY_TASK_ID)).pickle &> log/ridge_$((SLURM_ARRAY_TASK_ID)).log

# command for manual runnning
# python linear_regression.py --movie-offset 17700 --num-movies 100 --regr-type RandomForestRegressor --choosing-mechanism random_sample --max-reviews 1000 --num-cores 10 --ofile pickle/random_forest_results_10.pickle &> log/random_forest_10.log

