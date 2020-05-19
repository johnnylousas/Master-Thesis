import numpy as np
import pandas as pd
from Visualization import Visualization
from Process import Process
from Tuning import Tuning
from Classify import Classify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed()


def compute_data_drop(data, threshold: float):
    """
    Finds highly correlated columns (defined by threshold) and drops them, leaving the first one.
    :param data:
    :param threshold:
    :return: --- <class 'pandas.DataFrame'>
    """

    corr = data.corr()
    data_aux = data.copy()
    upper = corr.where(np.triu(corr, k=1).astype(np.bool))

    to_drop = [column for column in upper.columns if
               any(upper[column] > threshold) or any(upper[column] < -threshold)]

    return data_aux.drop(upper[to_drop], axis=1)


def rank(data):
    model = data.xgboost()[0]
    probs = model.predict_proba(data.valX)[:, 1]
    probs = np.round(probs * 100, 1)

    #create new column for the obtained probabilities
    dev['prob'] = probs

    #rank tests from higher to lower probability of failure
    Order = dev.sort_values(by=['prob'], ascending=False)

    #calculate APFD metric
    n = len(Order['class'])
    m = sum(Order['class'])
    pos = 0

    #APFD prioritize
    for i in range(len(Order['class'])):
        if Order['class'].values[i] == 1:
            pos += i
    apfd = 1 - pos/(n*m) + 1/(2*n)
    print(f'With Prioritisation: {apfd}')

    #APFD Random
    pos = 0
    rand = Order['class'].values

    np.random.shuffle(rand)
    for i in range(len(rand)):
        if rand[i] == 1:
            pos += i
    apfd_random = 1 - pos/(n*m) + 1/(2*n)
    print(f'Withou Prioritisation (Random): {apfd_random}')

    return apfd, apfd_random


class Data(Visualization, Process, Classify, Tuning):

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data.copy()
        self.y: np.ndarray = data.pop(target).values
        self.X: np.ndarray = data.values
        self.labels = pd.unique(self.y)
        self.trnX, self.tstX, self.trnY, self.tstY = train_test_split(self.X, self.y, train_size=0.8, stratify=self.y)
        self.trnX, self.valX, self.trnY, self.valY = train_test_split(self.trnX, self.trnY, test_size=0.125)



    def save_csv_metrics(self, dataset: str, prep_used: str):
        lr_metrics = self.logistic_regression()[1:]
        dt_metrics = self.decision_tree()[1:]
        rf_metrics = self.random_forest()[1:]
        xgb_metrics = self.xgboost()[1:]
        nn_metrics = self.ann()[1:]

        cols = ['Training_Loss', 'Training_Accuracy', 'ConfusionMatrix', 'Testing_Accuracy', 'Precision', 'Recall',
                'fScore', 'Support']
        my_dict = {
            "Logistic Regression": lr_metrics,
            "Decision Tree": dt_metrics,
            "Random Forest": rf_metrics,
            "Gradient Boosting": xgb_metrics,
            "Neural Network": nn_metrics
        }
        df = pd.DataFrame.from_dict(my_dict, orient='index', columns=cols)
        df.round()
        f = open('metrics/metrics_' + dataset + '.csv', 'w+')
        df.to_csv(f, sep='\t')
        f.write('# Dataset -' + dataset + '\n')
        f.write('# Preprocessing technique -' + prep_used + '\n')
        f.close()


if __name__ == '__main__':

    filename = '../data/tf_'

    train: pd.DataFrame = pd.read_csv(filename + 'train.csv')
    test: pd.DataFrame = pd.read_csv(filename + 'test.csv')
    dev: pd.DataFrame = pd.read_csv(filename + 'dev.csv')

    data = pd.concat([train, dev, test], ignore_index=False)
    data.rename(columns={'Unnamed: 0': 'test_id'}, inplace=True)
    # correlation analysis
    df = compute_data_drop(data=data.drop(columns=['test_id']), threshold=0.8)

    #date = data.pop('transactionid').values
    #date = data.pop('commitdate').values

    obj = Data(df, target='class')
    obj.minMax()
    prio, rand = rank(obj)

    #obj.save_csv_metrics(dataset='tempest_full', prep_used='MinMaxScaler')






