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

for number in {1..17700..1770}; do ./test.cmd $number; done
```
