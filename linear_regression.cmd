#!/bin/bash
# parallel job using 10 processors. and runs for 4 hours (max)
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=10
#SBATCH -t 4:00:00
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
python linear_regression.py --movie-offset $((1 + SLURM_ARRAY_TASK_ID*1770)) --num-movies 1770 --regr-type AdaBoostRegressor --choosing-mechanism random_sample --max-reviews 1000 --num-cores 10 --ofile pickle/ada_boost_results_$((SLURM_ARRAY_TASK_ID)).pickle &> log/ada_boost_$((SLURM_ARRAY_TASK_ID)).log

