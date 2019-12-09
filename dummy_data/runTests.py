import csv
import random

import pandas as pd
import numpy as np

src = pd.read_csv('src_files.csv')
test = pd.read_csv('test_files.csv')
commit = pd.read_csv('cmt_files.csv')

print("------------------------------------------------------------------------")
print('SOURCE LIST')
print("Data shape is: ", src.shape, "\t.....\t(instances, variables)")
print(src)
print("------------------------------------------------------------------------")


print("------------------------------------------------------------------------")
print('TEST LIST')
print("Data shape is: ", src.shape, "\t.....\t(instances, variables)")
print(test)
print("------------------------------------------------------------------------")


print("------------------------------------------------------------------------")
print('COMMIT LIST')
print("Data shape is: ", src.shape, "\t.....\t(instances, variables)")
print(commit)
print("------------------------------------------------------------------------")

print("------------------------------------------------------------------------")
# run test history contains the history of all tests that have been run.
print('RUN TEST HISTORY')
records = src.size()

fieldnames_runtesthistory = ['commit_id', 'test_id', 'timestamp', 'test_status', 'this_commit_estimated_status']
writer_runtesthistory = csv.DictWriter(open("rth_files.csv", "w"), fieldnames=fieldnames_runtesthistory)
writer_runtesthistory.writerow(dict(zip(fieldnames_runtesthistory, fieldnames_runtesthistory)))

for i in range(0, records):
    # Generates Source File list
    writer_runtesthistory.writerow(dict([
        ('commit_id', ),
        ('test_id', ),
        ('timestamp',),
        ('test_status',),
        ('this_commit_estimated_status', str(random.randint(0, 100)))]))

print("------------------------------------------------------------------------")


