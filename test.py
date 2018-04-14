import csv
from imdb import IMDb

ia = IMDb()

l = ['a','b', 'c']
movies = [['a', 'b', '+'.join(['a','b', 'c'])]]

ofile = open('data/test.csv', "w")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE, escapechar='\\')

for row in movies:
    writer.writerow(row)

ofile.close()

