import pickle, time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import scipy.sparse as sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# This file consists of titles and release years associated with each ID
movie_titles = pd.read_csv('data/COS424_hw3/movie_titles.txt', header = None, names = ['ID','Year','Name'], encoding='latin-1')
print movie_titles.head()
print movie_titles.shape

movie_by_id = {}
for id, name, year in zip(movie_titles['ID'], movie_titles['Name'], movie_titles['Year']):
    if not (np.isnan(year)):
        year = str(int(year))
    else:
        year = 'NaN'
    movie_by_id[id] = name + ' ' + '(' + year + ')'

# This file is a sparse matrix of movies by user, with each element a rating (1-5) or nonresponse (0)
ratings_csr = sparse.load_npz('data/COS424_hw3/netflix_full_csr.npz')
print ratings_csr.shape

# Filter the matrix to remove rows with NO REVIEWS
start = time.time()
ratings_csc = ratings_csr.T
print 'before removing users with no reviews: ', ratings_csc.shape
non_zero_users_csc = ratings_csc[(ratings_csc.getnnz(axis=1) != 0)]
print non_zero_users_csc.shape

finish = time.time()
print 'finished reduction in %.2f seconds' % (finish - start)
ratings_small = non_zero_users_csc

start = time.time()
svd = TruncatedSVD(n_components = 15, algorithm="arpack", random_state=0)
all_users_small = svd.fit_transform(ratings_small)
finish = time.time()
print all_users_small.shape
print 'finished svd in %.2f seconds' % (finish - start)


start = time.time()
kmeans_all_users = KMeans(n_clusters = 20 , random_state=0, algorithm="full", n_jobs=-1)
kmeans_all_users.fit(all_users_small)
finish = time.time()
print 'finished clustering in %.2f seconds' % (finish - start)
clusters_all_users = kmeans_all_users.labels_
clusters, counts = np.unique(clusters_all_users, return_counts=True)
print counts





with open('cluster_cache.pickle', 'w') as f:
    pickle.dump([kmeans_all_users, clusters_all_users, clusters, counts], f)

with open('svd_cache.pickle', 'w') as f:
    pickle.dump([svd, all_users_small], f)
