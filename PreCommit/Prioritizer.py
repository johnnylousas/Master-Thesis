import pandas as pd
from sklearn.manifold import TSNE
import umap

from Data import DataCI
from DataGenerator import DataGenerator
from Model import Model, Metrics
from Visualizer import Visualizer

from tensorflow import keras
import numpy as np


def reduce_dim(weights, components=3, method='tsne'):
    """Reduce dimensions of embeddings"""
    if method == 'tsne':
        return TSNE(components, metric='cosine').fit_transform(weights)
    elif method == 'umap':
        # Might want to try different parameters for UMAP
        return umap.UMAP(n_components=components, metric='cosine',
                    init='random', n_neighbors=5).fit_transform(weights)


class NNEmbeddings(Model, Metrics, Visualizer):
    """
    Neural Networks Embeddings model which inherits from abstract class Model and class Metrics.
    Once it is created, all the data becomes available from DataCI class and there is the possibility of loading a
    previusly trained model, or to train a new one
    """

    def __init__(self, D: DataCI, model_file: str = 'model.h5', load: bool = False):
        Model.__init__(self)
        Metrics.__init__(self)
        Visualizer.__init__(self)
        self.Data = D
        self.model_file = model_file
        self.nr_pairs = len(self.Data.pairs)

        if load:
            self.model = keras.models.load_model(self.model_file)
        else:
            self.model = self.build_model()
            self.train()

    def build_model(self, embedding_size=50, optimizer='Adam', classification=True):
        """
        Build model architecture/framework
        :return: model
        """

        from keras.layers import Input, Embedding, Dot, Reshape, Dense
        from keras.models import Model

        # Both inputs are 1-dimensional
        file = Input(name='file', shape=[1])
        test = Input(name='test', shape=[1])

        # Embedding the book (shape will be (None, 1, 50))
        file_embedding = Embedding(name='file_embedding',
                                   input_dim=len(self.Data.file_index),
                                   output_dim=embedding_size)(file)

        # Embedding the link (shape will be (None, 1, 50))
        test_embedding = Embedding(name='test_embedding',
                                   input_dim=len(self.Data.test_index),
                                   output_dim=embedding_size)(test)

        # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
        merged = Dot(name='dot_product', normalize=True, axes=2)([file_embedding, test_embedding])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape=[1])(merged)

        # If classification, add extra layer and loss function is binary cross entropy
        if classification:
            merged = Dense(1, activation='sigmoid')(merged)
            model = Model(inputs=[file, test], outputs=merged)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Otherwise loss function is mean squared error
        else:
            model = Model(inputs=[file, test], outputs=merged)
            model.compile(optimizer=optimizer, loss='mse')

        model.summary()
        return model

    def train(self, nb_epochs=15, n_positive=1000, negative_ratio=1, training_set_size=0.8, validation_set_size=0.1,
              save_model=False, plot=True):
        """
        Train model.
        :param plot: If true accuracy vs loss is plotted for training and validation set
        :param n_positive:
        :param negative_ratio:
        :param save_model:
        :param validation_set_size:
        :param training_set_size:
        :param nb_epochs:
        :return:
        """
        # Generate training set
        training_set = self.Data.pairs[:int(training_set_size * self.nr_pairs)]
        validation_set = self.Data.pairs[int(training_set_size * self.nr_pairs):int(training_set_size * self.nr_pairs) +
                                                                                int(
                                                                                    validation_set_size * self.nr_pairs)]

        train_gen = DataGenerator(pairs=training_set, pairs_set=set(self.Data.pairs),
                                  n_positive=n_positive, negative_ratio=negative_ratio)

        val_gen = DataGenerator(pairs=validation_set, pairs_set=set(self.Data.pairs),
                                n_positive=n_positive, negative_ratio=negative_ratio)

        # Train
        self.model = self.model.fit(train_gen,
                                    validation_data=val_gen,
                                    epochs=nb_epochs,
                                    verbose=2)
        if plot:
            self.plot_acc_loss(self.model)
        if save_model:
            self.model.save(self.model_file)

    def test(self, test_set_size=0.2):
        test_set = self.Data.pairs[int(test_set_size * self.nr_pairs):]
        n_positive = 1000

        test_gen = DataGenerator(pairs=test_set, pairs_set=set(self.Data.pairs),
                                 n_positive=n_positive, negative_ratio=1.0)

        pred = self.model.predict(test_gen, steps=len(test_set) // n_positive)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1

        return pred

    def predict(self, file_list: list, test_list: list):
        """
        Makes model prediction for unseen data.
        :param test_list:
        :param file_list:
        :return:
        """
        new_pairs = []
        for t in self.Data.all_tests:
            for f in file_list:
                new_pairs.append((self.Data.file_index[f], self.Data.test_index[t]))

        def generate_predictions(pairs, batch_size):
            batch = np.zeros((batch_size, 2))
            while True:
                for idx, (file_id, test_id) in enumerate(pairs):
                    batch[idx, :] = (file_id, test_id)

                # Increment idx by 1
                idx += 1

                yield {'file': batch[:, 0], 'test': batch[:, 1]}

        x = next(generate_predictions(new_pairs, len(file_list) * len(self.Data.all_tests)))

        new_pred = self.model.predict(x)

        p = zip(x['file'], x['test'])
        order = [x for _, x in sorted(zip(new_pred, p), reverse=True)]

        for idx, (file, test) in enumerate(order):
            if self.Data.index_test[test] in test_list:
                print(f'relevant test {self.Data.index_test[test]} at index - {idx}')

    def extract_weights(self, name):
        """Extract weights from a neural network model"""

        # Extract weights
        weight_layer = self.model.get_layer(name)
        weights = weight_layer.get_weights()[0]

        # Normalize
        weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
        return weights

    def get_components(self, method='TSNE'):
        file_weight_class = self.extract_weights('file_embedding')
        test_weight_class = self.extract_weights('test_embedding')
        file_r = reduce_dim(file_weight_class, components=2, method=method)
        test_r = reduce_dim(test_weight_class, components=2, method=method)

        return file_r, test_r

    def parameter_tuning(self):
        pass


def main():
    commits = pd.read_csv('../pub_data/test_commits_pub.csv', encoding='latin-1', sep='\t')
    test_details = pd.read_csv('../pub_data/test_details_pub.csv', sep='\t')
    test_status = pd.read_csv('../pub_data/test_histo_pub.csv', sep='\t')
    mod_files = pd.read_csv("../pub_data/test_commits_mod_files_pub.csv", sep='\t')

    D = DataCI(commits, test_details, test_status, mod_files)
    Prio = NNEmbeddings(D)



if __name__ == '__main__':
    main()
