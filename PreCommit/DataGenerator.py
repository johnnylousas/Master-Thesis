import random
import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras Model. Receives pairs of (files, tests) and generates random combinations of
    (file, test, label). Where the label is either 1 if the pair exists in the data or 0 otherwise. The class balance
    is given by the negative_ratio parameter, if parameter is 1, class balance is 50% each.
    Parameter splits gives the percentage attributed to (training_set, validation_set, test_set)
    """

    def __init__(self, pairs, pairs_set, n_positive=50, negative_ratio=1.0, classification=True, shuffle=True):
        """
        Data Generator constructor.
        :param pairs:
        :param pairs_set:
        :param n_positive:
        :param negative_ratio:
        :param classification:
        :param shuffle:
        """
        self.pairs = pairs
        self.pairs_set = pairs_set
        self.nr_files = len(set(item[0] for item in self.pairs))
        self.nr_tests = len(set(item[1] for item in self.pairs))

        self.n_positive = n_positive
        self.negative_ratio = negative_ratio
        self.batch_size = int(self.n_positive * (1 + self.negative_ratio))

        self.shuffle = shuffle
        self.classification = classification

        self.on_epoch_end()

    def __len__(self):
        """
        Gives length of one batch
        :return:
        """
        return int(np.floor(len(self.pairs) / self.batch_size))

    def on_epoch_end(self):
        """
        When epoch is finished shuffle indexes
        :return:
        """
        self.indexes = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Returns data generated in one batch
        :param index:
        :return:
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        pairs_temp = [self.pairs[k] for k in indexes]
        # Generate data
        X, y = next(self.__data_generation(pairs_temp))
        return X, y

    def __data_generation(self, pairs):
        """Generate batches of samples for training"""
        batch = np.zeros((self.batch_size, 3))

        # Adjust label based on task
        if self.classification:
            neg_label = 0
        else:
            neg_label = -1

        # This creates a generator
        while True:
            for idx, (file_id, test_id) in enumerate(random.sample(pairs, self.n_positive)):
                batch[idx, :] = (file_id, test_id, 1)

            # Increment idx by 1
            idx += 1

            # Add negative examples until reach batch size
            while idx < self.batch_size:

                # random selection
                random_file = random.randrange(self.nr_files)
                random_test = random.randrange(self.nr_tests)

                # Check to make sure this is not a positive example
                if (random_file, random_test) not in self.pairs_set:
                    # Add to batch and increment index
                    batch[idx, :] = (random_file, random_test, neg_label)
                    idx += 1

            np.random.shuffle(batch)
            yield {'file': batch[:, 0], 'test': batch[:, 1]}, batch[:, 2]
