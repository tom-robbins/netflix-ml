import csv

temp1 = open('data/temp.csv', "r")
temp2 = open('data/temp2.csv', "r")
temp3 = open('data/temp3.csv', "r")
temp4 = open('data/temp4.csv', "r")

movie_info = open('data/movie_info.csv', 'w')
writer = csv.writer(movie_info, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

for f in (temp1, temp2, temp3, temp4):
    reader = csv.reader(f, delimiter=',', quotechar='"', escapechar='\\')
    for row in reader:
        writer.writerow(row)

movie_info.close()
