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

sbatch linear_regression.cmd
```

## Baseline Regression Models
```sacct -u thomasrr --format=JobID,JobName,MaxRSS,Elapsed,exitcode```

### Linear Regression
```sacct -j 517450 --format=JobID,JobName,MaxRSS,Elapsed```
Used ~27889684K Memory

### Random Tree Regression
```sacct -j 517637 --format=JobID,JobName,MaxRSS,Elapsed```
Used ~47663916K Memory

### AdaBoost Regression
```sacct -j 517478 --format=JobID,JobName,MaxRSS,Elapsed```
Used ~47552932K Memory

### Ridge Regression
```sacct -j 518047 --format=JobID,JobName,MaxRSS,Elapsed```
Used ~48279576K Memory
