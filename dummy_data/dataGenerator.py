import csv
import random
from random import randrange
import datetime

import pandas as pd

records = 10
print("Making %d records\n" % records)

fieldnames_src = ['id', 'prevalence', 'stability']
fieldnames_test = ['id', 'time_to_run', 'dependence']
fieldnames_commits = ['commit_id', 'author', 'timestamp', 'modified_files', 'broken_files']

writer_src = csv.DictWriter(open("src_files.csv", "w"), fieldnames=fieldnames_src)
writer_test = csv.DictWriter(open("test_files.csv", "w"), fieldnames=fieldnames_test)
writer_commits = csv.DictWriter(open("cmt_files.csv", "w"), fieldnames=fieldnames_commits)

sources = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
people = ['Donnatelo', 'Rafael', 'Leonardo', 'Michelangelo']


# timestamp generator
def random_date(start, l):
    current = start
    while l >= 0:
        curr = current + datetime.timedelta(minutes=randrange(60))
        yield curr
        l -= 1


startDate = datetime.datetime(2013, 9, 20, 13, 00)
timestamps = []
for x in random_date(startDate, 10):
    timestamps.append(x.strftime("%d/%m/%y %H:%M"))
timestamps = sorted(timestamps)

writer_src.writerow(dict(zip(fieldnames_src, fieldnames_src)))
writer_test.writerow(dict(zip(fieldnames_test, fieldnames_test)))
writer_commits.writerow(dict(zip(fieldnames_commits, fieldnames_commits)))

for i in range(0, records):
    # Generates Test files list
    deps = random.sample(sources, k=random.randint(1, 4))
    writer_test.writerow(dict([
        ('id', i),
        ('time_to_run', str(random.randint(5, 60))),
        ('dependence', deps)]))

    # Generates Commit List
    cmt1 = random.sample(sources, k=random.randint(1, 4))
    writer_commits.writerow(dict([
        ('commit_id', 213 * i % 100 + 3 * i),
        ('author', random.choice(people)),
        ('timestamp', timestamps[i]),
        ('modified_files', cmt1),
        ('broken_files', random.sample(cmt1, k=random.randint(0, len(cmt1))))]))

tests = pd.read_csv('test_files.csv')
print(tests)
prevalence = writer_test['dependence'].count('a')

for i in range(0, records):
    # Generates Source File list
    writer_src.writerow(dict([
        ('id', sources[i]),
        ('prevalence', str(prevalence)),
        ('stability', str(random.randint(0, 100)))]))