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

