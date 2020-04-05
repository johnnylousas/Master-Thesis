import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier
import plotFunctions as func
import preprocess as prep


# Higly correlated feature elimination
from xgboost import XGBClassifier


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


class Classifier:
    def __init__(self, data: pd.DataFrame, target: str, datadrop: bool = False):

        if datadrop:
            data = compute_data_drop(data, threshold=0.9)
        self.data = data.copy()
        self.y: np.ndarray = data.pop(target).values
        self.X: np.ndarray = data.values
        self.labels = pd.unique(self.y)

        self.trnX, self.tstX, self.trnY, self.tstY = train_test_split(self.X, self.y, train_size=0.7, stratify=self.y)

    # Data Visualization
    def heatmap(self):
        fig = plt.figure(figsize=[12, 12])
        axs = plt.axes()
        corr_mtx = self.data.corr()
        func.draw_heatmap(ax=axs, data=corr_mtx)

    def missing_values(self):
        fig = plt.figure(figsize=(10, 7))
        mv = {}
        data = self.data
        for var in data:
            mv[var] = data[var].isna().sum()
            func.bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var,
                           'nr. missing values')
        fig.tight_layout()
        plt.show()

    def single_boxplots(self):
        self.data.boxplot(figsize=(10, 6))
        plt.show()

    def multiple_boxplots(self):
        columns = self.data.select_dtypes(include='number').columns
        rows, cols = func.choose_grid(len(columns))
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        i, j = 0, 0
        for n in range(len(columns)):
            axs[i, j].set_title('Boxplot for %s' % columns[n])
            axs[i, j].boxplot(data[columns[n]].dropna().values)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        fig.tight_layout()
        plt.show()

    def multiple_hist(self):
        columns = self.data.select_dtypes(include='number').columns
        rows, cols = func.choose_grid(len(columns))
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        i, j = 0, 0
        for n in range(len(columns)):
            axs[i, j].set_title('Histogram for %s' % columns[n])
            axs[i, j].set_xlabel(columns[n])
            axs[i, j].set_ylabel("probability")
            axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        fig.tight_layout()
        plt.show()

    def scatter_plot(self):
        columns = self.data.select_dtypes(include='number').columns
        rows, cols = len(columns) - 1, len(columns) - 1
        plt.figure()
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
        for i in range(len(columns)):
            var1 = columns[i]
            for j in range(i + 1, len(columns)):
                var2 = columns[j]
                axs[i, j - 1].set_title("%s x %s" % (var1, var2))
                axs[i, j - 1].set_xlabel(var1)
                axs[i, j - 1].set_ylabel(var2)
                axs[i, j - 1].scatter(data[var1], data[var2])
        fig.tight_layout()
        plt.show()

    def data_balance(self):
        target_count = self.data['bug'].value_counts()

        plt.figure()
        plt.title('Class balance')
        plt.bar(target_count.index, target_count.values)

        min_class = target_count.idxmin()
        ind_min_class = target_count.index.get_loc(min_class)

        print('Minority class:', target_count[ind_min_class])
        print('Majority class:', target_count[1 - ind_min_class])
        print('Proportion:', round(target_count[ind_min_class] / target_count[1 - ind_min_class], 2), ': 1')

        RANDOM_STATE = 42
        values = {'Original': [target_count.values[ind_min_class], target_count.values[1 - ind_min_class]]}

        df_class_min = self.data[self.data['bug'] == min_class]
        df_class_max = self.data[self.data['bug'] != min_class]

        df_under = df_class_max.sample(len(df_class_min))
        values['UnderSample'] = [target_count.values[ind_min_class], len(df_under)]

        df_over = df_class_min.sample(len(df_class_max), replace=True)
        values['OverSample'] = [len(df_over), target_count.values[1 - ind_min_class]]

        smote = SMOTE(ratio='minority', random_state=RANDOM_STATE)
        y = self.data.pop('bug').values
        X = self.data.values
        _, smote_y = smote.fit_sample(X, y)
        smote_target_count = pd.Series(smote_y).value_counts()
        values['SMOTE'] = [smote_target_count.values[ind_min_class], smote_target_count.values[1 - ind_min_class]]

        plt.figure()
        func.multiple_bar_chart(plt.gca(),
                                [target_count.index[ind_min_class], target_count.index[1 - ind_min_class]],
                                values, 'Target', 'frequency', 'Class balance')
        plt.show()

    # Data Preprocessing
    def preprocess(self, opt: str = None, k: int = 2):
        print('Preprocessing technique chosen', opt)
        if opt == 'minmax':
            self.trnX, self.tstX = prep.minMax(self.trnX, self.tstX)
        elif opt == 'standard':
            self.trnX, self.tstX = prep.standard(self.trnX, self.tstX)
        elif opt == 'smote':
            self.trnX, self.trnY = prep.smote(self.trnX, self.trnY)
            print('  shape %s' % str(self.trnX.shape))
        elif opt == 'nearmiss':
            self.trnX, self.trnY = prep.near_miss(self.trnX, self.trnY)
            print('  shape %s' % str(self.trn_x.shape))
        elif opt == 'undersample' or opt == 'outliers':
            self.trnX, self.trnY = prep.undersample(self.trnX, self.trnY)
        elif opt == 'pca':
            self.trnX, self.trnY, self.tst_x = prep.PrincipalComponentAnalysis(self.trnX, self.tstX, self.trnY, k)
        elif opt == 'lda':
            self.trnX, self.tstX = prep.LDA(self.trnX, self.tstX, self.trnY, k)
        elif opt == 'fs':
            self.trnX, self.tstX = prep.FS(self.trnX, self.tstX, self.trnY, k)
        else:
            print('# ============')
            print('  Option not available')
            print('  MinMaxScaler option:  minmax')
            print('  StandardScaler option:  standard')
            print('  SMOTE option:  smote')
            print('  Near Miss option:  nearmiss')
            print('  Under Sampling option:  undersample')
            print('  PCA option:  pca')
            print('  LDA option:  lda')
            print('  Feature Selection option:  fs')
            print('# ============')

    def PCA_param_tuning(self, model):
        comp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        yvalues = []
        axs = plt.axes()
        for i in comp:
            trn_x, tst_x = prep.PrincipalComponentAnalysis(self.trnX, self.tstX, self.trnY, k_value=i)
            model.fit(trn_x, self.trnY)
            prdY = model.predict(tst_x)
            yvalues.append(metrics.accuracy_score(self.tstY, prdY))
        func.scatter(ax=axs, xvalues=comp, yvalues=yvalues, xlabel='nr_components', ylabel='accuracy',
                     title='PCA_tuning')
        plt.show()

    def FS_param_tuning(self, model):
        comp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        yvalues = []
        axs = plt.axes()
        for i in comp:
            trn_x, tst_x = prep.FS(self.trnX, self.tstX, self.trnY, k_value=i)
            model.fit(trn_x, self.trnY)
            prdY = model.predict(tst_x)
            yvalues.append(metrics.accuracy_score(self.tstY, prdY))
        func.scatter(ax=axs, xvalues=comp, yvalues=yvalues, xlabel='nr_components', ylabel='accuracy',
                     title='FeatureSelection_tuning')
        plt.show()

    # Machine Learning Methods
    def logistic_regression(self, solver: str = 'liblinear', Cs: int = 10):
        print('#===============')
        print('Logistic Regression Model')
        model = LogisticRegression(solver=solver, C=Cs, random_state=42)
        model.fit(self.trnX, self.trnY)
        prdY = model.predict(self.tstX)

        accuracies = cross_val_score(estimator=model, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print('cross_validation')
        print('     ', 'mean', mean, 'variance', variance)

        yvalues = metrics.accuracy_score(self.tstY, prdY)
        print('test set accuracy')
        print('     ', yvalues)

        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        print('confusion matrix')
        print(cnf_mtx)

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

    def decision_tree(self, min_samples_leaf: float = 0.05, max_depth: int = 5, criterion: str = 'entropy'):
        print('#===============')
        print('Decision Tree Model')
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion,
                                      random_state=42)
        tree.fit(self.trnX, self.trnY)
        prdY = tree.predict(self.tstX)

        accuracies = cross_val_score(estimator=tree, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print('cross_validation')
        print('     ', 'mean', mean, 'variance', variance)

        yvalues = metrics.accuracy_score(self.tstY, prdY)
        print('test set accuracy')
        print('     ', yvalues)

        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        print('confusion matrix')
        print(cnf_mtx)

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

    def random_forest(self, maxdepth:int = 20, estim: int = 40, minsplit: float = 0.2, criterion: str = 'entropy'):
        print('#===============')
        print('Random Forest Model')
        rf = RandomForestClassifier(max_depth=maxdepth, random_state=41, max_features='sqrt',
                                    min_samples_split=minsplit, n_estimators=estim, criterion=criterion)
        rf.fit(self.trnX, self.trnY)
        prdY = rf.predict(self.tstX)

        accuracies = cross_val_score(estimator=rf, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print('cross_validation')
        print('     ', 'mean', mean, 'variance', variance)

        yvalues = metrics.accuracy_score(self.tstY, prdY)
        print('test set accuracy')
        print('     ', yvalues)

        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        print('confusion matrix')
        print(cnf_mtx)

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

    def xgboost(self, booster: str = 'gbtree'):
        print('#===============')
        print('Gradient Boosting Model')
        xgb = XGBClassifier(booster=booster)
        xgb.fit(self.trnX, self.trnY)
        prdY = xgb.predict(self.tstX)

        accuracies = cross_val_score(estimator=xgb, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print('cross_validation')
        print('     ', 'mean', mean, 'variance', variance)

        yvalues = metrics.accuracy_score(self.tstY, prdY)
        print('test set accuracy')
        print('     ', yvalues)

        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        print('confusion matrix')
        print(cnf_mtx)

    def ann(self):
        # Importing the Keras libraries and packages
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout

        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=15))
        classifier.add(Dropout(p=0.1))

        # Adding the second hidden layer
        classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dropout(p=0.1))

        # Adding the output layer
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(self.trnX, self.trnY, batch_size=10, epochs=100)

        # Part 3 - Making predictions and evaluating the model

        # Predicting the Test set results
        y_pred = classifier.predict(self.tstX)
        y_pred = (y_pred > 0.5)

    def eval_ann(self):

        # Evaluating the ANN
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import cross_val_score
        from keras.models import Sequential
        from keras.layers import Dense

        def build_classifier():
            classifier = Sequential()
            classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=15))
            # classifier.add(Dropout(p=0.1))
            classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
            # classifier.add(Dropout(p=0.1))
            classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return classifier

        classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
        accuracies = cross_val_score(estimator=classifier, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print('cross_validation')
        print('     ', 'mean', mean, 'variance', variance)

    def improve_ann(self):
        # Improving the ANN
        # Dropout Regularization to reduce overfitting if needed

        # Tuning the ANN
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import GridSearchCV
        from keras.models import Sequential
        from keras.layers import Dense

        def build_classifier():
            # parameter tuning
            classifier = Sequential()
            classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=15))
            classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return classifier

        classifier = KerasClassifier(build_fn=build_classifier)
        parameters = {'batch_size': [10, 25],
                      'epochs': [100, 500]
                      }
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=parameters,
                                   scoring='accuracy',
                                   cv=10)
        grid_search = grid_search.fit(self.trnX, self.trnY)
        best_parameters = grid_search.best_params_
        best_accuracy = grid_search.best_score_
        print(best_parameters, best_accuracy)


# =========

# Data prep
data: pd.DataFrame = pd.read_csv('input/bugzilla.csv')
date = data.pop('commitdate').values

# Class element creation
classifier = Classifier(data, target='bug', datadrop=False)

# Desired Preprocessing
classifier.preprocess(opt='minmax')
classifier.preprocess(opt='fs', k=2)

# Model Application
#classifier.logistic_regression()
#classifier.decision_tree()
#classifier.random_forest()
#classifier.xgboost()

classifier.improve_RF()

# classifier.ann()
# classifier.eval_ann()
# classifier.improve_ann()
# best params (batch size: 10 , epochs: 500) long computation time , already done
