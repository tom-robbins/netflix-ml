import csv
import re
from imdb import IMDb

ia = IMDb()

genres = {}
people = {}
movies = []

tempfile = open('data/temp4.csv', "w")
tempwriter = csv.writer(tempfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

with open('data/movies4.txt', 'r', encoding="ISO-8859-1") as movie_titles:
    for line in movie_titles:
        l = line.split(',')
        num = l[0]
        year = l[1]
        title = (',').join(l[2:])
        cast = []
        genres = []

        s_result = ia.search_movie(l[2])
        if s_result:
            item = s_result[0]
            ia.update(item)
            try:
                cast = [person.personID for person in item['cast']]
            except:
                pass
            try:
                genres = item['genres']
            except:
                pass
            print(num, item['long imdb canonical title'])
        else:
            print('COULDNT FIND ', title.strip())

        l = [num, year, re.sub('\'\",', '', title).strip(), '+'.join(cast), '+'.join(genres)]
        l = [i if i else 'NaN' for i in l]
        movies.append(l)
        tempwriter.writerow(l)

ofile  = open('data/movie_info.csv', "w")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

for row in movies:
    writer.writerow(row)

ofile.close()

