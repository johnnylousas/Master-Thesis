import numpy as np
import pandas as pd
from Visualization import Visualization
from Process import Process
from Tuning import Tuning
from Classify import Classify
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class Data(Visualization, Process, Classify, Tuning):

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data.copy()
        self.y: np.ndarray = data.pop(target).values
        self.X: np.ndarray = data.values
        self.labels = pd.unique(self.y)
        self.trnX, self.tstX, self.trnY, self.tstY = train_test_split(self.X, self.y, train_size=0.7, stratify=self.y)

    def save_csv_metrics(self, dataset: str, prep_used: str):

        lr_metrics = self.logistic_regression()
        dt_metrics = self.decision_tree()
        rf_metrics = self.random_forest()
        xgb_metrics = self.xgboost()
        nn_metrics = self.ann()

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
    filename = 'postgres'

    data: pd.DataFrame = pd.read_csv('input/' + filename + '.csv')
    date = data.pop('commitdate').values
    obj = Data(data, target='bug')
    obj.minMax()
    obj.undersample()
    obj.logistic_regression()
    obj.xgboost()
    #obj.save_csv_metrics(dataset=filename, prep_used='MinMax & undersampling')