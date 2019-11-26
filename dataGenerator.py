import pandas as pd
import numpy as np
import csv
import random

records = 10
print("Making %d records\n" % records)

fieldnames_src = ['id', 'prevalence', 'stability']
fieldnames_test = ['id', 'time', 'dependence']
fieldnames_commits = ['id', 'who', 'what_files', 'broken_files']

writer_src = csv.DictWriter(open("src_files.csv", "w"), fieldnames=fieldnames_src)
writer_test = csv.DictWriter(open("test_files.csv", "w"), fieldnames=fieldnames_test)
writer_commits = csv.DictWriter(open("cmt_files.csv", "w"), fieldnames=fieldnames_commits)

sources = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
people = ['Donnatelo', 'Rafael', 'Leonardo', 'Michelangelo']

# filling src_files.csv
writer_src.writerow(dict(zip(fieldnames_src, fieldnames_src)))
for i in range(0, records):
    writer_src.writerow(dict([
        ('id', sources[i]),
        ('prevalence', str(random.randint(0, 100))),
        ('stability', str(random.randint(0, 100)))]))

# filling test_files.csv
writer_test.writerow(dict(zip(fieldnames_test, fieldnames_test)))
for i in range(0, records):
    writer_test.writerow(dict([
        ('id', i),
        ('time', str(random.randint(5, 60))),
        ('dependence', random.choices(sources, k=random.randint(1, 4)))]))

# filling cmt_files.csv
writer_commits.writerow(dict(zip(fieldnames_commits, fieldnames_commits)))
for i in range(0, records):
    cmt1 = random.sample(sources, k=random.randint(1, 4))
    writer_commits.writerow(dict([
        ('id', i),
        ('who', random.choice(people)),
        ('what_files', cmt1),
        ('broken_files', random.sample(cmt1, k=random.randint(0, len(cmt1))))]))




