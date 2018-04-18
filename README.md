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
```
source scraping/bin/activate
pip install imdbpy numpy scikit-learn surprise
```
The scraping files are info_scraper.py, stitch.py, and onehot.py one-hot encodes them for regression.  The scraper has a small chance of scraping incorrect information for movies, as it is based on the search functionality of IMDb, and for 'movies' in the dataset that are really just Bonus Content, TV series', concerts, or Special Editions, the search with their full name string often doesn't produce results.  The recursive searching only affects ~6% of the movies, and brings the number of genre/actor information gained from ~87% to ~93%

### Using the adroit cluster
```
ssh thomasrr@adroit.princeton.edu
module load anaconda
conda create --name netflix numpy scipy pandas matplotlib scikit-learn
source activate netflix

# make sure to change the parameters in the .cmd file before sending it to SLURM
sbatch linear_regression.cmd
```
we used the SLURM system on the adroit cluster to run the linear regression models and mean-based models, using ssh to access and sftp to transfer files.

### Model
linear_regression.py runs the entirety of the prediction process, usage:
```
python linear_regression.py --movie-offset <int> --num-movies <int> [--regr-type <LinearRegression, Ridge, RandomForestRegressor, AdaBoostRegressor>] [--cluster] [--include-user-average] --choosing-mechanism <random_sample, top_reviewers> --max-reviews <int> --num-cores <int> --ofile <path> &> <path>
```
the --cluster flag runs the clustering, and with the --include_user_average it runs the Clus+ model
the --regr-type chooses the type of regression used in the regression models

### Python Notebooks
We did not transfer

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

### CLuster Means
```518094```
Used ~52791712K Memory

### Cluster Linear Regression


### Residuals
518290

### with user aavg
518343

