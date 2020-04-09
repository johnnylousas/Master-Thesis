import matplotlib.pyplot as plt
import plotFunctions as func
import preprocess as preprocess
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score


class Tuning:

    def __init__(self):
        pass

    def improve_LogReg(self, nr_folds: int = 10):
        C = [1, 2, 3, 5, 10, 15]
        solver = ['liblinear', 'newton-cg', 'lbfgs']

        axs = plt.axes()
        values = {}
        for k in range(len(solver)):
            f = solver[k]
            yvalues = []
            for i in C:
                model = LogisticRegressionCV(cv=nr_folds, solver=f, Cs=i, random_state=0)
                model.fit(self.trnX, self.trnY)
                prdY = model.predict(self.tstX)
                yvalues.append(metrics.accuracy_score(self.tstY, prdY))
            values[f] = yvalues
            func.multiple_line_chart(axs, C, values, xlabel='REG', ylabel='accuracy', title='LogisticRegression ',
                                     percentage=True)

        plt.show()

    def improve_DT(self, nr_folds: int = 10):
        min_samples_leaf = [.05, .025, .01, .0075, .005, .0025, .001]
        max_depths = [5, 10, 25, 50]
        criteria = ['entropy', 'gini']

        fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
        for k in range(len(criteria)):
            f = criteria[k]
            values = {}
            for d in max_depths:
                yvalues = []
                for n in min_samples_leaf:
                    tree = DecisionTreeClassifier(min_samples_leaf=n, max_depth=d, criterion=f)
                    dt_fit = tree.fit(self.trnX, self.trnY)
                    prdY = tree.predict(self.tstX)
                    scores = cross_val_score(dt_fit, self.trnX, self.trnY, cv=nr_folds)
                    yvalues.append(scores.mean())
                values[d] = yvalues
            func.multiple_line_chart(axs[0, k], min_samples_leaf, values, 'Decision Trees with %s criteria' % f,
                                     'nr estimators',
                                     'accuracy', percentage=True)
        plt.show()

    def improve_RF(self, nr_folds: int = 10):
        n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
        max_depths = [5, 10, 25, 50]
        max_features = ['sqrt', 'log2']

        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
        for k in range(len(max_features)):
            f = max_features[k]
            values = {}
            for d in max_depths:
                yvalues = []
                for n in n_estimators:
                    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
                    dt_fit = rf.fit(self.trnX, self.trnY)
                    prdY = rf.predict(self.tstX)
                    scores = cross_val_score(dt_fit, self.trnX, self.trnY, cv=nr_folds)
                    yvalues.append(scores.mean())
                values[d] = yvalues
            func.multiple_line_chart(axs[0, k], n_estimators, values, 'Random Forests with %s features' % f,
                                     'nr estimators',
                                     'accuracy', percentage=True)

        plt.show()

    def pca_param_tuning(self, model):
        comp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        yvalues = []
        axs = plt.axes()
        for i in comp:
            trn_x, tst_x = preprocess.pComponentAnalysis(self.trnX, self.tstX, self.trnY, k_value=i)
            model.fit(trn_x, self.trnY)
            prdY = model.predict(tst_x)
            yvalues.append(metrics.accuracy_score(self.tstY, prdY))
        func.scatter(ax=axs, xvalues=comp, yvalues=yvalues, xlabel='nr_components', ylabel='accuracy',
                     title='PCA_tuning')
        plt.show()

    def fs_param_tuning(self, model):
        comp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        yvalues = []
        axs = plt.axes()
        for i in comp:
            trn_x, tst_x = preprocess.fs(self.trnX, self.tstX, self.trnY, k_value=i)
            model.fit(trn_x, self.trnY)
            prdY = model.predict(tst_x)
            yvalues.append(metrics.accuracy_score(self.tstY, prdY))
        func.scatter(ax=axs, xvalues=comp, yvalues=yvalues, xlabel='nr_components', ylabel='accuracy',
                     title='FeatureSelection_tuning')
        plt.show()