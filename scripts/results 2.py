import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

np.random.seed()
"""
filename = 'data/tf_'

train: pd.DataFrame = pd.read_csv(filename + 'train.csv')
test: pd.DataFrame = pd.read_csv(filename + 'test.csv')
dev: pd.DataFrame = pd.read_csv(filename + 'dev.csv')

data = pd.concat([train, dev, test], ignore_index=False)
data.rename(columns={'Unnamed: 0': 'test_id'}, inplace=True)
data = data.drop(columns=['test_id'])

y: np.ndarray = data.pop('class').values
X: np.ndarray = data.values
"""

ClosureCompiler = pd.read_csv('pred-rep-master/tanzeem_noor-promise17_data/Closure-Compiler Metrics Raw_Data.csv')
ClosureCompiler = ClosureCompiler.drop(columns=['TestID', 'TestName', 'Status'])

y: np.ndarray = ClosureCompiler.pop('Result').values
X: np.ndarray = ClosureCompiler.values
labels = pd.unique(y)


def rank(clf):
    kf = KFold(n_splits=10, random_state=42, shuffle=True)

    accuracies = []
    apfd_sort = []
    apfd_rand = []

    for train_index, test_index in kf.split(X):
        trnX = X[train_index]
        trnY = y[train_index]

        tstX = X[test_index]
        tstY = y[test_index]

        # if needed, do preprocessing here
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(trnX)
        trnX = scaler.transform(trnX)
        tstX = scaler.transform(tstX)

        # clf = LogisticRegression(solver='liblinear')
        # clf = XGBClassifier()

        clf.fit(trnX, trnY)

        preds = clf.predict(tstX)
        # accuracy for the current fold only
        accuracy = accuracy_score(tstY, preds)
        accuracies.append(accuracy)

        probs = clf.predict_proba(tstX)[:, 1]
        probs = np.round(probs * 100, 1)

        nr_rand = random.sample(range(len(tstY)), len(tstY))
        test_order = [x for _, x in sorted(zip(probs, tstY), reverse=True)]
        test_random = [x for _, x in sorted(zip(nr_rand, tstY), reverse=True)]

        n = len(tstY)
        m = sum(tstY)
        pos = 0

        for i in range(n):
            if test_order[i] == 1:
                pos += i
        apfd = 1 - pos / (n * m) + 1 / (2 * n)
        apfd_sort.append(apfd)

        # APFD Random
        pos = 0
        for i in range(n):
            if test_random[i] == 1:
                pos += i
        apfd_random = 1 - pos / (n * m) + 1 / (2 * n)
        apfd_rand.append(apfd_random)

    return apfd_sort, apfd_rand


models = [LogisticRegression(solver='liblinear'), DecisionTreeClassifier(), RandomForestClassifier(), XGBClassifier()]
apfds = []
i = 0
for clf in models:
    apfd_sort = rank(clf)[0]
    apfds.append(apfd_sort)
    i += 1
    if i == len(models):
        apfds.append(rank(clf)[1])

plt.figure()
plt.boxplot(apfds, labels=['LR', 'DT', 'RF', 'XGB', 'Random'])
plt.title('Test Case Ranking Performance' + ' ' + 'Closure-Compiler')
plt.ylabel('APFD')
plt.ylim((0, 1))
plt.show()
