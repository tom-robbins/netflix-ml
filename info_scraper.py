import csv
import re
from imdb import IMDb

ia = IMDb()

genres = {}
people = {}
movies = []

tempfile = open('data/failed1.csv', "w")
tempwriter = csv.writer(tempfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

season = re.compile(r'(Season \d)|(Seasons \d+ & \d+)')
volume = re.compile(r'Vols?.? \d')
bonus = re.compile(r'Bonus Material')
edition = re.compile(r':.*?Edition')
anniversary = re.compile(r'\d+.*Anniversary:? (Edition|Collection|Tour|Double Feature|Specials)?')

count = 0
with open('data/failed_movies.csv', 'r', encoding="ISO-8859-1") as movie_titles:
    for line in movie_titles:
        print(count)
        count += 1

        l = line.split(',')
        num = l[0]
        year = l[1]
        title = (',').join(l[2:])
        cast = []
        genres = []

        s_result = None
        string_to_search = l[2]

        words_cut_off = 0
        title_length = len(title.split())

        # Remove special text pieces from the titles
        if re.search(anniversary, string_to_search):
            string_to_search = re.sub(anniversary, '', string_to_search)
        # if its a TV show, strip the season number
        # if its bonus material, strip that off the end too
        for regex in [edition, season, volume, bonus]:
            result = re.search(regex, string_to_search)
            if result:
                string_to_search = result.group(0).join(string_to_search.split(result.group(0))[:-1])
                if string_to_search[-1] == ':':
                    string_to_search = string_to_search[:-1]

        s_result = ia.search_movie(string_to_search)

        # cut off last words?
        # while(string_to_search and not s_result):
        #     s_result = ia.search_movie(string_to_search)
        #     if s_result:
        #         print(string_to_search)
        #         continue

        #     if words_cut_off > 2 or title_length < 4:
        #         print('TOO MANY CUTOFFS FOR: %s' % string_to_search)
        #         break
        #     string_to_search = ' '.join(string_to_search.split(' ')[:-1])
        #     words_cut_off += 1


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

ofile  = open('data/failed_movie_info.csv', "w")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL, escapechar='\\')

for row in movies:
    writer.writerow(row)

ofile.close()

