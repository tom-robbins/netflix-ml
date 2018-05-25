#!/bin/bash
# parallel job using 10 processors. and runs for 2 hours (max)
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=10
#SBATCH -t 2:00:00
#SBATCH --array=0-9
#SBATCH --job-name=cluster_means_vanilla
#SBATCH --mem 80000
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
python linear_regression.py --movie-offset $((1 + SLURM_ARRAY_TASK_ID*1800)) --num-movies 1800 --regr-type LinearRegression --cluster --choosing-mechanism random_sample --max-reviews 1000 --num-cores 10 --ofile pickle/vanilla_cluster_results_$((SLURM_ARRAY_TASK_ID)).pickle &> log/vanilla_cluster_$((SLURM_ARRAY_TASK_ID)).log

# command for manual runnning
# python linear_regression.py --movie-offset 17700 --num-movies 100 --regr-type LinearRegression --cluster --choosing-mechanism random_sample --max-reviews 1000 --num-cores 10 --ofile pickle/cluster_means_results_10.pickle &> log/cluster_means_10.log


