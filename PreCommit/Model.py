from abc import abstractmethod
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class Model:
    """
    Abstract class to define which model will serve as a classifier for data.
    """

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, nb_epoch, batch_size, training_set_size, validation_set_size):
        pass

    @abstractmethod
    def test(self, test_set_size, ):
        pass

    @abstractmethod
    def predict(self, file_list: list, test_list: list):
        pass

    @abstractmethod
    def parameter_tuning(self):
        pass


class Metrics:
    """
    Implements Metrics Class for classification evaluation.
    """

    def accuracy(self, y_true, y_pred):
        """
        Returns accuracy score metric
        :param y_true:
        :param y_pred:
        :return: acc
        """
        return accuracy_score(y_true, y_pred)

    def cnf_mtx(self, y_true, y_pred):
        """
        Determines the confusion matrix for a given prediction.
        :param y_true:
        :param y_pred:
        :return: confusion_matrix
        """
        return confusion_matrix(y_true, y_pred)

    def report(self, y_true, y_pred):
        """
        Give Classification Report
        :param y_true:
        :param y_pred:
        :return:
        """
        return classification_report(y_true, y_pred)

    def apfd(self, y_true, y_pred):
        """
        Calculates the Average Percentage of Fault Detection. Given a prioritiation the APFD is near 1, if all relevant
        tests are applied at the beginning and 0 otherwise.
        :param y_true:
        :param y_pred:
        :return: apfd
        """
        pass

    def fdr(self, y_true, y_pred, duration):
        """
        Fault detection rate: y-axis -> percentage of faults detected - x-axis -> percentage of time spent
        :param duration:
        :param y_true:
        :param y_pred:
        :return: fdr
        """
        pass

    def pretty_print_stats(self):
        """
        returns report of statistics for a given model object
        """
        items = (('accuracy:', self.accuracy()), ('sst:',),
                 ('mse:',), ('r^2:',),
                 ('adj_r^2:',))

        for item in items:
            print('{0:8} {1:.4f}'.format(item[0], item[1]))


