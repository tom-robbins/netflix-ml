import csv

movie_info = open('data/movie_info.csv', 'r')
onehot = open('data/onehot_movie_info.csv', 'w')

reader = csv.reader(movie_info, delimiter=',', quotechar='"', escapechar='\\')
writer = csv.writer(onehot, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE, escapechar='\\')

total_genres = set()
total_people = set()
movie_dict = {}

yearnan = 0
castnan = 0
genrenan = 0

for row in reader:
    if row[1] == 'NULL':
        year = 'NaN'
        yearnan += 1
        print(row)
    else:
        year = int(row[1])

    if row[3] == 'NaN':
        cast = ['NaN']
        castnan += 1
    else:
        cast = list(set([int(i) for i in row[3].split('+')]))
        for p in cast:
            total_people.add(p)

    if row[4] == 'NaN':
        genres = ['NaN']
        genrenan += 1
    else:
        genres = row[4].split('+')
        for g in genres:
            total_genres.add(g)

    movie_dict[int(row[0])] = [year, row[2], cast, genres]

print(movie_dict[1])
print(len(total_people))

print('years nan', yearnan)
print('genres nan', genrenan)
print('casts nan', castnan)

# Make the one-hot-encoded csv for genre data

genrelist = list(total_genres)
writer.writerow(['movieID'] + genrelist)
for i in range(1, len(movie_dict) + 1):
    row = [i] + [1 if g in movie_dict[i][3] else 0 for g in genrelist]
    writer.writerow(row)


