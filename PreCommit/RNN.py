from sklearn.model_selection import KFold, StratifiedKFold
from keras.utils import to_categorical

from Data import DataCI
from Model import Model, Metrics
from Visualizer import Visualizer
import numpy as np


class RNN(Model, Metrics, Visualizer):
    """
    Neural Networks Embeddings model which inherits from abstract class Model and class Metrics.
    Once it is created, all the data becomes available from DataCI class and there is the possibility of loading a
    previusly trained model, or to train a new one
    """

    def __init__(self, D: DataCI):
        """
        NNEmbeddings Class initialization.
        :param D:
        :param model_file:
        :param embedding_size:
        :param optimizer:
        :param save:
        :param load:
        """
        Model.__init__(self)
        Metrics.__init__(self)
        Visualizer.__init__(self)
        self.Data = D
        self.Data.df_link = self.Data.df_link.explode('name')
        print(self.Data.df_link[['mod_files', 'name']])

        file_sequence = self.Data.df_link['mod_files'].tolist()
        test_sequence = self.Data.df_link['name'].tolist()
        print(len(file_sequence))
        print(len(test_sequence))

        file_sequence = [[self.Data.file_index[file] for file in seq] for seq in file_sequence]
        test_sequence = [self.Data.test_index[test] for test in test_sequence]
        test_sequence = to_categorical(test_sequence, num_classes=len(self.Data.test_index))

        max_len = max([len(seq) for seq in file_sequence])

        self.model = self.build_model(max_len)
        self.model.fit(file_sequence, test_sequence, validation_split=0.2, verbose=1, epochs=10)
        exit()

        self.crossValidation(X=file_sequence, y=test_sequence)

    def build_model(self, embedding_size=50, optimizer='Adam', max_len=10):
        """
        Build model architecture/framework
        :return: model
        """

        from keras import Sequential
        from keras.layers import Embedding, Dropout, LSTM, Dense
        from keras.models import Model

        model = Sequential()
        model.add(Embedding(input_dim=len(self.Data.file_index), output_dim=embedding_size, input_length=max_len - 1))
        model.add(Dropout(0.2))
        model.add(LSTM(3))
        model.add(Dropout(0.2))
        model.add(Dense(len(self.Data.test_index), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        model.summary()
        return model

    def crossValidation(self, X, y,  k_folds=10, nb_epochs=10, save_model=False):
        # Training with K-fold cross validation
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

        X = np.array(X)
        y = np.array(y)

        cv_accuracy_train = []
        cv_accuracy_val = []
        cv_loss_train = []
        cv_loss_val = []

        i = 1
        for train_index, test_index in kf.split(X, y):

            print("=========================================")
            print("====== K Fold Validation step => %d/%d =======" % (i, k_folds))
            print("=========================================")

            # Train
            h = self.model.fit(X[train_index], y[train_index],
                               epochs=nb_epochs,
                               verbose=2)

            pred = self.model.predict(X[test_index], y[test_index])

            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1

            cv_accuracy_train.append(np.array(h.history['accuracy'])[-1])
            cv_accuracy_val.append(np.array(h.history['val_accuracy'])[-1])
            cv_loss_train.append(np.array(h.history['loss'])[-1])
            cv_loss_val.append(np.array(h.history['val_loss'])[-1])

            i += 1

        if save_model:
            self.model.save(self.model_file)

        return np.mean(cv_accuracy_train), np.mean(cv_loss_train), \
               np.mean(cv_accuracy_val), np.mean(cv_loss_val)