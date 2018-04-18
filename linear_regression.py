import argparse, math, multiprocessing, os, pickle, sys, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, csc_matrix
from scipy import spatial
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB



# Handle the command line arguments for running the linear regressions
parser = argparse.ArgumentParser()
parser.add_argument('--movie-offset',dest='movie_offset', type=int, help='index of first movie to model (1-17770)')
parser.add_argument('--num-movies', dest='num_movies', type=int, help='number of movies to model (17770 total)')
parser.add_argument('--regr-type', dest='regr_type', help='regression algorithm (LinearRegression, AdaBoostRegressor, RandomForestRegressor, Ridge)')
parser.add_argument('--cluster', dest='perform_cluster', action='store_true', help='perform KMeans clustering on the data or not (used in the clustering models)')
parser.add_argument('--include-user-average', dest='include_user_average', action='store_true', help='add user average delta to the predictions for mean-based model')
parser.add_argument('--choosing-mechanism', dest='choosing_mechanism', help='how to choose features (random_sample, top_reviewers)')
parser.add_argument('--max-reviews', dest='max_reviews', type=int, help='max number of features (reviews) for regression')
parser.add_argument('--num-cores', dest='num_cores', type=int, help='number of cores to use')
parser.add_argument('--ofile', dest='ofile', help='path to the output file (pickled dictionary)')


args = parser.parse_args(sys.argv[1:])



# This file consists of titles and release years associated with each ID
movie_titles = pd.read_csv('data/movie_titles.txt', header = None, names = ['ID','Year','Name'])
# print movie_titles.head()
print 'shape of of titles:', movie_titles.shape

movie_by_id = {}
for id, name, year in zip(movie_titles['ID'], movie_titles['Name'], movie_titles['Year']):
    if not (np.isnan(year)):
        year = str(int(year))
    else:
        year = 'NaN'
    movie_by_id[id] = name + ' ' + '(' + year + ')'



# import the movie genre data scraped using imdbpy
movie_genres = pd.read_csv('data/onehot_all_movie_genres.csv', header = 0)
print 'shape of of genres:', movie_genres.shape



# This file is a sparse matrix of movies by user, with each element a rating (1-5) or nonresponse (0)
ratings_csr = sparse.load_npz('data/COS424_hw3/netflix_full_csr.npz')
print 'shape of of ratings:', ratings_csr.shape
print



# Filter the matris to remove rows with NO REVIEWS
start = time.time()
ratings_csc = ratings_csr.T
print 'ratings shape before removing users with no reviews:', ratings_csc.shape
non_zero_users_csc = ratings_csc[(ratings_csc.getnnz(axis=1) != 0)]
print 'ratings shape after removing users with no reviews:', non_zero_users_csc.shape

finish = time.time()
print 'finished in %.2f seconds' % (finish - start)
print



# construct a dictionary to store number of reviews per user
non_zero_users_csr = csr_matrix(non_zero_users_csc)

reviews_by_user = {}
for u in range(non_zero_users_csr.shape[0]):
    reviews_by_user[u] = non_zero_users_csr[u].nnz


# s = sorted(reviews_by_user.keys(), key=lambda x: reviews_by_user[x], reverse=True)[:10]
# print 'highest amount of reviews per user:', [reviews_by_user[i] for i in s]
# print [i for i in s]



# count the number of reviews for each film and store in review_nums list
review_nums = []
for i in range(non_zero_users_csc.shape[1]):
    num_reviews = non_zero_users_csc[:,i].nnz
    review_nums.append((i, num_reviews, np.sum(non_zero_users_csc[:,i]) / num_reviews))

# Print the top movies by number of reviews
s = sorted(review_nums, key=lambda x: x[2])
print '#revs\tavg.\tmovie'
for movie_id, num, avg_review in s[-20:]:
    print '%s\t%0.4f\t%s' % (num, avg_review, movie_by_id[movie_id])
print


# # analyze the one guy who has seen 17,563 movies (WTF)
# hasnt_watched = []
# last = 0
# for i in sparse.find(non_zero_users_csr[55373])[1]:
#     if i - last > 1:
#         for j in range(1, i - last)[::-1]:
#             hasnt_watched.append(i - j)
#     last = i

# # sort by average star rating
# hasnt_watched = sorted(hasnt_watched, key=lambda x: review_nums[x][2], reverse=True)



# transpose back to (movie, user) orientation for effcient operations later
ratings_small = non_zero_users_csc
ratings_small = ratings_small.transpose()

# Add in rows for each genre
arr = np.array(movie_genres)
arr = np.insert(arr, 0, 0, axis=0)  # add zero row to match the zero row in ratings_small
csr_arr = csr_matrix(arr[:,1:])     # chop off the movie IDs and make it a csr matrix
ratings_small_with_genres = sparse.hstack([ratings_small, csr_arr], format = 'csr')  # append



# MAIN REGRESSION CELL

max_reviews = args.max_reviews                  # how many features are allowed in the regression
movie_results_dict = {}                         # destination dict for metrics/results
# top_reviewers, random_sample
choosing_mechanism = args.choosing_mechanism    # choose how to pick the features
data = ratings_small_with_genres                # choose the matrix to regress on

# pointers to the indexes of the genre features (not review data) appended at the end
end_genre_index = ratings_small_with_genres.shape[1]
num_genres = 27
beg_genre_index = end_genre_index - num_genres
genre_indexes = range(beg_genre_index, end_genre_index)

def run_regression(first_movie, last_movie, results_dict, perform_regression=False, include_user_average=False):
    # Loop over movie IDs, generating a model for each movie
    for movie_id in range(first_movie, last_movie):
        if movie_id < 1 or movie_id > data.shape[0] - 1:
            continue

        # keep track of what movie you're on
        num_reviews = data[movie_id,:beg_genre_index].count_nonzero()
        movie_name = movie_by_id[movie_id]
        print 'Movie #%s, %s\naverage rating: %.2f in %i reviews  | ' % (
            movie_id,
            movie_name,
            np.sum(data[movie_id,:beg_genre_index]) / num_reviews,
            int(num_reviews)
        ),

        ### PART I.
        ### MAKE THE X, y TO FEED THE REGRESSION OUT OF THE REVIEW DATA
        start = time.time()

        # filter out the reviewers who havent seen the movie
        user_mask = sparse.find(data[movie_id,:beg_genre_index])[1]

        # if there are many reviewers, only take some (based on choosing_mechanism)
        if (num_reviews > max_reviews):
            if (choosing_mechanism == 'top_reviewers'):
                user_mask = sorted(user_mask, key=lambda u: reviews_by_user[u])[:max_reviews]
            elif (choosing_mechanism == 'random_sample'):
                user_mask = user_mask[np.random.choice(len(user_mask), size=max_reviews)]

        # Include the genre features
        #user_mask = np.unique(np.hstack([genre_indexes, user_mask]))

        print 'regressing on %i features' % len(user_mask)

        # apply mask to filter the users (axis 0) of the X matrix
        ratings_filtered_by_user = data[:,user_mask]

        # make mask to filter out only the reviews for the movie in question
        movie_mask = np.ravel(np.full((ratings_filtered_by_user.shape[0], 1), True))
        movie_mask[movie_id] = False

        # generate X and y, training and testing data splitting
        X = ratings_filtered_by_user[movie_mask]
        y = ratings_filtered_by_user[movie_id]
        X_train, X_test, y_train, y_test = train_test_split(X.transpose().todense(), np.ravel(y.transpose().todense()), test_size=0.2, random_state=10)

        finish = time.time()
        data_time = (finish - start)

        ### PART II
        ### RUN THE REGRESSION ON THE FILM TO MAKE THE PREDICTOR
        start = time.time()

        # Set the regression based on the type argument
        if args.regr_type == 'RandomForestRegressor':
            regr = RandomForestRegressor(n_estimators=50)
        elif args.regr_type == 'Ridge':
            regr = Ridge(alpha=0.1)
        elif args.regr_type == 'AdaBoostRegressor':
            regr = AdaBoostRegressor()
        else:
            regr = LinearRegression()

        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)

        current_mse = mean_squared_error(y_test, y_pred)
        current_r2 = r2_score(y_test, y_pred)


        finish = time.time()
        regr_time = (finish - start)

        # add data to results dictionary
        results_dict[movie_id] = {
            'name': movie_by_id[movie_id],
            'id': movie_id,
            'regr_time': regr_time,
            'data_time': data_time,
            #'regr': regr,
            'mse': current_mse,
            'r2': current_r2,
        }

        print "MSE: %.2f \t\t r2: %.2f \t\t time: %.2f ... (%.2f + %.2f)" % (
            current_mse,
            current_r2,
            data_time + regr_time, data_time, regr_time
        )
        print



# Only used for the clustering models, so only perform (expensive) calculations if necessary
if args.perform_cluster:
    # Run SVD on the dataset
    if os.path.isfile('cache/svd_cache.pickle'):
        with open('cache/svd_cache.pickle', 'r') as f:
            [svd, all_users_small] = pickle.load(f)
        print 'found svd cache file'
    else:
        start = time.time()
        svd = TruncatedSVD(n_components = 5, algorithm="arpack", random_state=0)
        all_users_small = svd.fit_transform(ratings_small)
        finish = time.time()
        print(all_users_small.shape)
        print('finished svd in %.2f seconds' % (finish - start))

    # Cluster using KMeans based on the SVD output for users
    if os.path.isfile('cache/cluster_cache.pickle'):
        with open('cache/cluster_cache.pickle', 'r') as f:
            [kmeans_all_users, clusters_all_users, clusters, counts] = pickle.load(f)
        print 'found cluster cache file'
    else:
        raise
        start = time.time()
        kmeans_all_users = KMeans(n_clusters = 20 , random_state=0, algorithm="full")
        kmeans_all_users.fit(all_users_small)
        finish = time.time()
        print('finished clustering in %.2f seconds' % (finish - start))
        clusters_all_users = kmeans_all_users.labels_
        clusters, counts = np.unique(clusters_all_users, return_counts=True)
        print(counts)

ratings_csr = csr_matrix(non_zero_users_csc)

# User average helper functions
def user_avg_without_movie(user_id, movie_id):
    sums = ratings_csr[user_id,:movie_id].sum() + ratings_csr[user_id,movie_id+1:].sum()
    nums = ratings_csr[user_id,:movie_id].getnnz() + ratings_csr[user_id,movie_id+1:].getnnz()
    return sums / nums

def cluster_avg_without_user(cluster_id, user_id):
    sums = ratings_csr[clusters_all_users == cluster_id].sum() - ratings_csr[user_id].sum()
    nums = ratings_csr[clusters_all_users == cluster_id].getnnz() - ratings_csr[user_id].getnnz()
    return sums / nums

# find the average movie review
rating_average = ratings_csr.sum() / ratings_csc.getnnz()
summ = 0
# find the average movie review per cluster
cluster_averages = []
for cluster in clusters:
    clust_data = ratings_csr[clusters_all_users == cluster]
    cluster_averages.append(clust_data.sum() / clust_data.getnnz())
    summ += clust_data.getnnz()




# Clustering model creation
# either makes a mean-based model
# or a regression based model (regress over the reviews of a cluster of users)
def run_cluster_means(first_movie, last_movie, results_dict, perform_regression=False, include_user_average=False):
    for movie_id in range(first_movie, last_movie):
        if movie_id < 1 or movie_id > data.shape[0] - 1:
            continue
        y_pred_mode = []
        y_pred_mean = []
        y_test_all = []
        num_reviews = non_zero_users_csc[:,movie_id].count_nonzero()
        movie_name = movie_by_id[movie_id]
        print 'Movie #%s, %s\naverage rating: %.2f in %i reviews  | ' % (
            movie_id,
            movie_name,
            np.sum(data[movie_id,:beg_genre_index]) / num_reviews,
            int(num_reviews)
        ),

        # REGRESSION OR NAH
        if perform_regression:
            start = time.time()
            filter_by = np.ravel((non_zero_users_csc[:,movie_id] != 0.0).toarray())
            filtered_clusters = clusters_all_users[filter_by]
            filtered_ratings = non_zero_users_csc[filter_by,:]

            movie_mask = np.ravel(np.full((filtered_ratings.shape[1], 1), True))
            movie_mask[movie_id] = False
            X = filtered_ratings[:,movie_mask]
            y = filtered_ratings[:,movie_id].toarray()

            X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, filtered_clusters, test_size=0.2, random_state=0)

            finish = time.time()
            data_time = finish - start
            start = time.time()

            clusters_for_this_movie = np.unique(c_test)
            regrs = []
            total_y_pred = []
            total_y_test = []

            for cluster in clusters_for_this_movie:
                # Set the regression based on the type argument
                if args.regr_type == 'RandomForestRegressor':
                    regr = RandomForestRegressor(n_estimators=50)
                elif args.regr_type == 'Ridge':
                    regr = Ridge(alpha=0.1)
                elif args.regr_type == 'AdaBoostRegressor':
                    regr = AdaBoostRegressor()
                else:
                    regr = LinearRegression()

                new_X_train = X_train[c_train==cluster]
                new_y_train = y_train[c_train==cluster]
                if new_y_train.shape[0] == 0:
                    new_X_train = X_train
                    new_y_train = y_train
                if new_X_train.shape[0] > max_reviews:
                    random_mask = np.random.choice(new_X_train.shape[0], size=max_reviews, replace=False)
                    new_X_train = new_X_train[random_mask]
                    new_y_train = new_y_train[random_mask]

                regr.fit(new_X_train.toarray(), new_y_train)
                regrs.append(regr)

            for i, cluster in enumerate(clusters_for_this_movie):
                new_X_test = X_test[c_test==cluster]
                new_y_test = y_test[c_test==cluster]
                new_y_pred = regrs[i].predict(new_X_test.toarray())
                total_y_pred = np.append(total_y_pred, new_y_pred)
                total_y_test = np.append(total_y_test, new_y_test)

            # compute metrics
            current_r2 = r2_score(total_y_test, total_y_pred)
            current_mse = mean_squared_error(total_y_test, total_y_pred)
        elif include_user_average:
            start = time.time()
            movie_mask = np.ravel(np.full((non_zero_users_csc.shape[1], 1), True))
            movie_mask[movie_id] = False

            filter_by = np.ravel((non_zero_users_csc[:,movie_id] != 0.0).toarray())
            filtered_clusters = clusters_all_users[filter_by]
            filtered_ratings = non_zero_users_csc[filter_by,:]

            indexes = np.ravel(np.where(filter_by)[0])

            X = filtered_ratings
            y = filtered_ratings[:,movie_id].toarray()

            X_train, X_test, y_train, y_test, c_train, c_test, i_train, i_test = train_test_split(X, y, filtered_clusters, indexes, test_size=0.2, random_state=0)


            finish = time.time()
            data_time = finish - start
            start = time.time()

            y_pred_mean = []
            y_test_all = []

            mean_all = np.mean(y_train)
            cluster_predictions = [
                np.mean(y_train[c_train == cluster]) if len(y_train[c_train == cluster]) else None for cluster in clusters
            ]



            for (index, cluster, review) in zip(i_test, c_test, y_test):
                y_test_all.append(review)
                this_user_avg = user_avg_without_movie(index, movie_id)
                if cluster_predictions[cluster] is not None:
                    if np.isnan(this_user_avg):
                        this_user_avg = cluster_averages[cluster]
                    user_difference = this_user_avg - cluster_averages[cluster]
                    y_pred_mean.append(cluster_predictions[cluster] + user_difference)
                else:
                    if np.isnan(this_user_avg):
                        this_user_avg = mean_all
                    user_difference = this_user_avg - mean_all
                    y_pred_mean.append(mean_all + user_difference)

            # compute metrics
            current_r2 = r2_score(y_test_all, y_pred_mean)
            current_mse = mean_squared_error(y_test_all, y_pred_mean)
        else:
            start = time.time()
            filter_by = np.ravel((non_zero_users_csc[:,movie_id] != 0.0).toarray())
            filtered_clusters = clusters_all_users[filter_by]
            filtered_ratings = non_zero_users_csc[filter_by,:]

            movie_mask = np.ravel(np.full((filtered_ratings.shape[1], 1), True))
            movie_mask[movie_id] = False
            X = filtered_ratings[:,movie_mask]
            y = filtered_ratings[:,movie_id].toarray()

            X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(X, y, filtered_clusters, test_size=0.2, random_state=0)

            finish = time.time()
            data_time = finish - start
            start = time.time()

            for (cluster, review) in zip(c_test, y_test):
                y_test_all.append(review)
                cluster_reviews = y_train[c_train == cluster]
                mean_all = np.mean(y_train)

                if len(cluster_reviews) != 0:
                    y_pred_mean.append(np.mean(cluster_reviews))
                else:
                    y_pred_mean.append(mean_all)

            # compute metrics
            current_r2 = r2_score(y_test_all, y_pred_mean)
            current_mse = mean_squared_error(y_test_all, y_pred_mean)


        finish = time.time()
        regr_time = finish - start

        results_dict[movie_id] = {
            'name': movie_by_id[movie_id],
            'id': movie_id,
            'regr_time': regr_time,
            'data_time': data_time,
            'mse': current_mse,
            'r2': current_r2,
        }
        print "MSE: %.2f \t\t r2: %.2f \t\t time: %.2f ... (%.2f + %.2f)" % (
            current_mse,
            current_r2,
            data_time + regr_time, data_time, regr_time
        )
        print



# Multiprocessing Code
multistart = time.time()
manager = multiprocessing.Manager()
movie_results_dict = manager.dict()
processes = []

# pass in arguments to the manager to start processes running the linear regressions
movie_offset = args.movie_offset
num_movies = args.num_movies
num_processes = args.num_cores
movies_per_process = num_movies / num_processes

# choose function to run to form models based on --cluster argument
if args.perform_cluster:
    target_function = run_cluster_means
else:
    target_function = run_regression

# start num_processes Processes, each running the regression on movies_per_process
# movies, and storing the results in movie_results_dict
for i in range(num_processes):
    p = multiprocessing.Process(
        target=target_function,
        args=(
            movie_offset + i*movies_per_process,
            movie_offset + (i+1)*movies_per_process,
            movie_results_dict,
        ),
        kwargs={
            'perform_regression': False,
            'include_user_average': args.include_user_average,
        }
    )
    p.start()
    processes.append(p)

# wait for all processes to finish
for p in processes:
    p.join()

multifinish = time.time()

print 'finished all regs in %.4f' % (multifinish - multistart)


# cursory analytics
def average_nested_dict_key(d, k):
    sum = np.sum([d[i][k] for i in d.keys()])
    return sum / len(d.keys())


ks = movie_results_dict.keys()
# sort by r2 value
print 'r2'
s = sorted(ks, key=lambda x: movie_results_dict[x]['r2'], reverse=True)
for i in s[:10]:
    print '%.3f\t%s' % (movie_results_dict[i]['r2'], movie_results_dict[i]['name'])

print
print 'average r2 value: %.2f' % average_nested_dict_key(movie_results_dict, 'r2')

# sort by mse
print '\nMSE'
s = sorted(ks, key=lambda x: movie_results_dict[x]['mse'])
for i in s[:10]:
    print '%.3f\t%s' % (movie_results_dict[i]['mse'], movie_results_dict[i]['name'])

print
print 'average MSE value: %.2f' % average_nested_dict_key(movie_results_dict, 'mse')



# scratch space for pickling data structures for safe keeping
with open(args.ofile, 'wb') as f:
    pickle.dump(dict(movie_results_dict), f)
