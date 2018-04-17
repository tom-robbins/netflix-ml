# Netflix Prize
COS 424 Assignment 3

## Caterina Golner & Tom Robbins
```
cgolner
thomasrr
```

## Guide

### scraping genre data
use scraping virtualenv and install python libraries
```console
source scraping/bin/activate
pip install imdbpy numpy scikit-learn surprise
```

### using the adroit cluster
```
ssh thomasrr@adroit.princeton.edu
module load anaconda
conda create --name netflix numpy scipy pandas matplotlib scikit-learn
source activate netflix

# make sure to change the parameters in the .cmd file before sending it to SLURM
sbatch linear_regression.cmd
```

## Baseline Regression Models
```sacct -u thomasrr --format=JobID,JobName,MaxRSS,Elapsed,exitcode```

### Linear Regression
```517450```
Used ~27889684K Memory

### Random Tree Regression
```517637```
Used ~47663916K Memory

### AdaBoost Regression
```517478```
Used ~47552932K Memory

### Ridge Regression
```518047```
Used ~48279576K Memory
