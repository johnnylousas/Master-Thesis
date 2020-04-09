from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
import numpy as np
from xgboost import XGBClassifier
from tabulate import tabulate

np.random.seed(0)


class Classify:

    def __init__(self):
        pass

    def logistic_regression(self, solver: str = 'liblinear', Cs: int = 10):
        print('#===============')
        model = LogisticRegression(solver=solver, C=Cs, random_state=42)
        model.fit(self.trnX, self.trnY)
        prdY = model.predict(self.tstX)

        accuracy = cross_val_score(estimator=model, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        trn_acc = accuracy.mean()

        loss = cross_val_score(estimator=model, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1, scoring='neg_log_loss')
        trn_loss = loss.mean()

        tst_acc = metrics.accuracy_score(self.tstY, prdY)
        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        precision, recall, fscore, support = precision_recall_fscore_support(self.tstY, prdY, average='macro')

        print(
            tabulate([['Logistic Regression', trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]],
                     headers=['Model',
                              'Training_Loss', 'Training_Accuracy', 'ConfusionMatrix', 'Testing_Accuracy', 'Precision',
                              'Recall',
                              'fScore', 'Support'], tablefmt='orgtbl'))

        return [trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]

    def decision_tree(self, min_samples_leaf: float = 0.05, max_depth: int = 5, criterion: str = 'entropy'):
        print('#===============')
        tree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, max_depth=max_depth, criterion=criterion,
                                      random_state=42)
        tree.fit(self.trnX, self.trnY)
        prdY = tree.predict(self.tstX)

        accuracy = cross_val_score(estimator=tree, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        trn_acc = accuracy.mean()

        loss = cross_val_score(estimator=tree, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1, scoring='neg_log_loss')
        trn_loss = loss.mean()

        tst_acc = metrics.accuracy_score(self.tstY, prdY)
        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        precision, recall, fscore, support = precision_recall_fscore_support(self.tstY, prdY, average='macro')

        print(
            tabulate([['Decision Tree', trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]],
                     headers=['Model',
                              'Training_Loss', 'Training_Accuracy', 'ConfusionMatrix', 'Testing_Accuracy', 'Precision',
                              'Recall',
                              'fScore', 'Support'], tablefmt='orgtbl'))

        return [trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]

    def random_forest(self, maxdepth: int = 20, estim: int = 40, minsplit: float = 0.2, criterion: str = 'entropy'):
        print('#===============')
        rf = RandomForestClassifier(max_depth=maxdepth, random_state=41, max_features='sqrt',
                                    min_samples_split=minsplit, n_estimators=estim, criterion=criterion)
        rf.fit(self.trnX, self.trnY)
        prdY = rf.predict(self.tstX)

        accuracy = cross_val_score(estimator=rf, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        trn_acc = accuracy.mean()

        loss = cross_val_score(estimator=rf, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1, scoring='neg_log_loss')
        trn_loss = loss.mean()

        tst_acc = metrics.accuracy_score(self.tstY, prdY)
        precision, recall, fscore, support = precision_recall_fscore_support(self.tstY, prdY, average='macro')
        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)

        print(
            tabulate([['Random Forest', trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]],
                     headers=['Model',
                              'Training_Loss', 'Training_Accuracy', 'ConfusionMatrix', 'Testing_Accuracy', 'Precision',
                              'Recall',
                              'fScore', 'Support'], tablefmt='orgtbl'))

        return [trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]

    def xgboost(self, booster: str = 'gbtree'):
        print('#===============')
        xgb = XGBClassifier(booster=booster)
        xgb.fit(self.trnX, self.trnY)
        prdY = xgb.predict(self.tstX)

        accuracy = cross_val_score(estimator=xgb, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        trn_acc = accuracy.mean()

        loss = cross_val_score(estimator=xgb, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1, scoring='neg_log_loss')
        trn_loss = loss.mean()

        tst_acc = metrics.accuracy_score(self.tstY, prdY)
        cnf_mtx = metrics.confusion_matrix(self.tstY, prdY, self.labels)
        precision, recall, fscore, support = precision_recall_fscore_support(self.tstY, prdY, average='macro')

        print(
            tabulate([['Gradient Boosting', trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]],
                     headers=['Model',
                              'Training_Loss', 'Training_Accuracy', 'ConfusionMatrix', 'Testing_Accuracy', 'Precision',
                              'Recall',
                              'fScore', 'Support'], tablefmt='orgtbl'))
        return [trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]

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

        trn_loss, trn_acc = classifier.evaluate(self.trnX, self.trnY, batch_size=10)
        # Part 3 - Making predictions and evaluating the model

        # Predicting the Test set results
        y_pred = classifier.predict(self.tstX)
        y_pred = (y_pred > 0.5)

        tst_acc = metrics.accuracy_score(self.tstY, y_pred)
        cnf_mtx = metrics.confusion_matrix(self.tstY, y_pred, self.labels)

        precision, recall, fscore, support = precision_recall_fscore_support(self.tstY, y_pred, average='macro')
        print(
            tabulate([['ANN', trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]],
                     headers=['Model',
                              'Training_Loss', 'Training_Accuracy', 'ConfusionMatrix', 'Testing_Accuracy', 'Precision',
                              'Recall',
                              'fScore', 'Support'], tablefmt='orgtbl'))
        return [trn_loss, trn_acc, cnf_mtx, tst_acc, precision, recall, fscore, support]

    def ann_CV(self):
        # Evaluating the ANN
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import cross_val_score
        from keras.models import Sequential
        from keras.layers import Dense

        def build_classifier():
            classifier = Sequential()
            classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu', input_dim=15))
            classifier.add(Dropout(p=0.1))
            classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
            classifier.add(Dropout(p=0.1))
            classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
            classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return classifier

        classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
        accuracies = cross_val_score(estimator=classifier, X=self.trnX, y=self.trnY, cv=10, n_jobs=-1)
        mean = accuracies.mean()
        variance = accuracies.std()
        print('cross_validation')
        print('     ', 'mean', mean, 'variance', variance)
        return mean, variance
