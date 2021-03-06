import numpy as np
import pandas as pd
import copy
from time import time
from typing import *
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.stats import chisquare
from prettytable import PrettyTable
from project_libs import ColorizedLogger
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from project_libs import timeit
from project_libs.project.plotter import plot_bpnn_results
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
import warnings
import pickle
from project_libs.project import one_hot_unencode
import xgboost as xgb

logger = ColorizedLogger('Models', 'green')

np.seterr(divide='raise')


# Implementation ofCase 1, 2, and 3 Bayesian
class BayesianCase:
    """ Implementation of Minimum Euclidean distance, Mahalanobis, and Quadratic classifiers. """

    mtypes: Tuple[str] = ("euclidean", "mahalanobis", "quadratic")
    g_builders: Dict[str, Callable] = dict.fromkeys(mtypes, [])
    accuracy: Dict[str, float]
    classwise_accuracy: Dict[str, List]
    prediction_time: Dict[str, float]
    predicted_y: Dict[str, np.ndarray]
    means: np.ndarray
    stds: np.ndarray
    covs: np.ndarray
    avg_mean: np.ndarray
    avg_std: np.ndarray
    first_and_second_case_cov: np.ndarray
    avg_var: np.ndarray
    tp: Dict[str, int]
    fn: Dict[str, int]
    fp: Dict[str, int]
    tn: Dict[str, int]

    def __init__(self, train: np.ndarray = None,
                 train_x: np.ndarray = None, train_y: np.ndarray = None,
                 test: np.ndarray = None,
                 test_x: np.ndarray = None, test_y: np.ndarray = None) -> None:
        # Initializations
        self.g_builders = {self.mtypes[0]: self._build_g_euclidean,
                           self.mtypes[1]: self._build_g_mahalanobis,
                           self.mtypes[2]: self._build_g_quadratic}
        self.classwise_accuracy = dict.fromkeys(self.mtypes, [])
        self.predicted_y = dict.fromkeys(self.mtypes, None)
        self.accuracy = dict.fromkeys(self.mtypes, None)
        self.prediction_time = dict.fromkeys(self.mtypes, None)
        self.tp = dict.fromkeys(self.mtypes, None)
        self.fn = dict.fromkeys(self.mtypes, None)
        self.fp = dict.fromkeys(self.mtypes, None)
        self.tn = dict.fromkeys(self.mtypes, None)
        # Separate features and labels from train and test set
        if train is not None:
            self.x_train, self.y_train = self.x_y_split(train)
        elif train_x is not None and train_y is not None:
            self.x_train, self.y_train = train_x, train_y
        else:
            raise Exception("You should either train or train_x and train_y!")
        if test is not None:
            self.x_test, self.y_test = self.x_y_split(test)
        elif test_x is not None and test_y is not None:
            self.x_test, self.y_test = test_x, test_y
        else:
            raise Exception("You should either train or train_x and train_y!")
        # Find the # of samples, features and classes
        self.n_samples_train, self.n_features = self.x_train.shape
        self.n_samples_test = self.x_test.shape[0]
        # Unique values (classes) of the features column
        self.unique_classes = np.unique(self.y_train).astype(int)

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1].astype(int)

    def fit(self) -> None:
        """ Trains the model on the training dataset and returns the means and the average variance """
        # Calculate means, covariance for each feature
        means = []
        stds = []
        covs = []
        for class_n in self.unique_classes:
            x_train_current_class = self.x_train[self.y_train == self.unique_classes[class_n]]
            means.append(x_train_current_class.mean(axis=0))
            stds.append(x_train_current_class.std(axis=0))
            covs.append(np.cov(x_train_current_class.T))
        # Calculate average covariance and variance
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.covs = np.array(covs)
        self.avg_mean = np.mean(self.means, axis=0)
        self.avg_std = np.mean(self.stds, axis=0)

    def _build_g_euclidean(self, sample, n_class, priors: List[float]):
        first_term = np.matmul(self.means[n_class].T, self.x_test[sample]) / self.avg_var
        second_term = np.matmul(self.means[n_class].T, self.means[n_class]) / (2 * self.avg_var)
        third_term = np.log(priors[n_class])
        g = first_term - second_term + third_term
        return g

    def _build_g_mahalanobis(self, sample, n_class, priors: List[float]):
        current_cov = self.first_and_second_case_cov
        try:
            first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                         np.linalg.inv(current_cov))
        except np.linalg.LinAlgError as e:
            logger.debug(f"{e}")
            current_cov += + 10e-5
            if str(e).strip() == 'Singular matrix':
                first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                             np.linalg.pinv(current_cov))
            else:
                current_cov += + 10e-5
                first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                             np.linalg.inv(current_cov))

        first_term = -(1 / 2) * np.matmul(first_term_dot_1,
                                          (self.x_test[sample] - self.means[n_class]))
        second_term = np.log(priors[n_class])
        g = first_term + second_term
        return g

    def _build_g_quadratic(self, sample, n_class, priors: List[float]):
        current_covs = np.abs(self.covs[n_class] + 10e-5)
        try:
            first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                         np.linalg.inv(current_covs))
        except np.linalg.LinAlgError as e:
            logger.debug(f"{e}")
            if str(e).strip() == 'Singular matrix':
                first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                             np.linalg.pinv(current_covs))
            else:
                current_covs += + 10e-5
                first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                             np.linalg.inv(current_covs))
        except Exception as e:
            logger.debug(f"{e}")
            first_term_dot_1 = (self.x_test[sample] - self.means[n_class]).T / current_covs

        first_term = -(1 / 2) * np.matmul(first_term_dot_1,
                                          (self.x_test[sample] - self.means[n_class]))
        try:
            second_term = -(1 / 2) * np.log(np.abs(np.linalg.det(current_covs) + 10e-5))
        except Exception as e:
            logger.debug(f"{e}")
            second_term = -(1 / 2) * np.log(np.abs(current_covs) + 10e-5)
        third_term = np.log(priors[n_class])
        g = first_term + second_term + third_term
        return g

    def predict(self, mtype: str, priors: List[float] = None,
                first_and_second_case_cov_type: str = 'avg',
                save_data: bool = False,
                extra_name: str = '') -> np.ndarray:
        """ Tests the model on the test dataset and returns the accuracy. """

        # Which covariance to use in the first and second case
        if first_and_second_case_cov_type == 'avg':
            self.first_and_second_case_cov = np.mean(self.covs, axis=0)
        elif first_and_second_case_cov_type == 'first':
            self.first_and_second_case_cov = self.covs[0]
        elif first_and_second_case_cov_type == 'second':
            self.first_and_second_case_cov = self.covs[1]
        else:
            raise Exception('first_and_second_case_cov_type should be one of: avg, first, second')
        # Calculate avg_var based on the choice
        try:
            self.avg_var = np.mean(np.diagonal(self.first_and_second_case_cov), axis=0)
        except ValueError as e:
            logger.warning(f"{e}")
            self.avg_var = self.first_and_second_case_cov
        # If no priors were given, set them as equal
        if not priors:
            priors = [1.0 / len(self.unique_classes) for _ in self.unique_classes]
        # Determine the model type and get correct function for building the g
        assert mtype in self.mtypes
        build_g = self.g_builders[mtype]
        # Predict the values
        start = time()
        _predicted_y = []
        for sample in range(self.n_samples_test):
            g = np.zeros(len(self.unique_classes))
            for n_class in self.unique_classes:
                # Calculate g for each class and append to a list
                g[n_class] = build_g(sample=sample, n_class=n_class, priors=priors)
            _predicted_y.append(g.argmax())
        self.predicted_y[mtype] = np.array(_predicted_y)
        self.prediction_time[mtype] = time() - start
        if save_data:
            self.save_pickle(self.predicted_y[mtype][:, np.newaxis],
                             case=mtype, extra_name=extra_name)
        return self.predicted_y[mtype]

    def get_statistics(self, mtype: str) -> Tuple[float, List[float], float]:
        """ Return the statistics of the model """
        # Check if mtype exists
        assert mtype in self.mtypes
        # Calculate metrics
        self.accuracy[mtype] = np.count_nonzero(self.predicted_y[mtype] == self.y_test) / len(
            self.predicted_y[mtype])
        self.classwise_accuracy[mtype] = []
        for class_n in self.unique_classes:
            y_test_current = self.y_test[self.y_test == self.unique_classes[class_n]]
            predicted_y_current = self.predicted_y[mtype][self.y_test == self.unique_classes[class_n]]
            current_acc = np.count_nonzero(predicted_y_current == y_test_current) / len(
                predicted_y_current)
            self.classwise_accuracy[mtype].append(current_acc)

        return self.accuracy[mtype], self.classwise_accuracy[mtype], self.prediction_time[mtype]

    def get_confusion_matrix(self, mtype: str) -> Tuple[int, int, int, int]:
        # Get True Positives
        y_test_positive = self.y_test[self.y_test == self.unique_classes[0]]
        y_pred_positive = self.predicted_y[mtype][self.y_test == self.unique_classes[0]]
        self.tp[mtype] = np.count_nonzero(y_pred_positive == y_test_positive)
        # Get False Positives
        self.fn[mtype] = np.count_nonzero(y_pred_positive != y_test_positive)
        # Get True Negatives
        y_test_negative = self.y_test[self.y_test == self.unique_classes[1]]
        y_pred_negative = self.predicted_y[mtype][self.y_test == self.unique_classes[1]]
        self.tn[mtype] = np.count_nonzero(y_test_negative == y_pred_negative)
        # Get False Negatives
        self.fp[mtype] = np.count_nonzero(y_test_negative != y_pred_negative)
        # Error Checking
        # from sklearn.metrics import confusion_matrix
        # print(confusion_matrix(self.y_test, self.predicted_y[mtype]))
        # print(np.array([[self.tp[mtype], self.fn[mtype]], [self.fp[mtype], self.tn[mtype]]]))
        return self.tp[mtype], self.fn[mtype], self.fp[mtype], self.tn[mtype]

    def print_statistics(self, name: str, mtype: str) -> None:
        # Check if statistics have be calculated
        if any(v is None for v in [self.accuracy, self.classwise_accuracy, self.prediction_time]):
            self.get_statistics(mtype)
        logger.info(f"Parametric Model (case: {mtype}) for the {name} dataset")
        logger.info(f"The overall accuracy is: {self.accuracy[mtype]:.4f}")
        logger.info(f"The classwise accuracies are: {self.classwise_accuracy[mtype]}")
        logger.info(f"Total time: {self.prediction_time[mtype]:.4f} sec(s)")
        logger.info(f"|{'':^15}|{'Positive':^15}|{'Negative':^15}|", color='red')
        logger.info(f"|{'Positive':^15}|{self.tp[mtype]:^15}|{self.fn[mtype]:^15}|", color='red')
        logger.info(f"|{'Negative':^15}|{self.fp[mtype]:^15}|{self.tn[mtype]:^15}|", color='red')

    @staticmethod
    def save_pickle(var: Any, case: str, extra_name: str = ''):
        path = f'data/bayesian/bayesian_{case}_case_{extra_name}_predicted.pickle'
        with open(path, 'wb') as handle:
            pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Logistic Regression Algorithm
class Log_Reg:

    def __init__(self, learning_rate, iters):
        self.learning_rate = learning_rate
        self.iters = iters
        self.weights, self.bias = None, None

    def predict(self, X, threshold):
        linear_pred = (np.dot(X, self.weights) + self.bias)
        probabilities = 1 / (1 + np.exp(-1 * linear_pred))
        return [1 if i > threshold else 0 for i in probabilities]

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for i in range(self.iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            probability = 1 / (1 + np.exp(-1 * linear_pred))
            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, (probability - y)))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(probability - y))

            self.weights -= self.learning_rate * partial_w
            self.bias -= self.learning_rate * partial_d

    def F1_score_func(self, actual, pred):
        self.cm = confusion_matrix(actual, pred)
        accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()
        precision = self.cm[1, 1] / (self.cm[1, 1] + self.cm[0, 1])
        sensitivity = self.cm[1, 1] / (self.cm[1, 1] + self.cm[1, 0])
        F1_Score = (2 * precision * sensitivity) / (precision + sensitivity)
        self.F1_Score = F1_Score
        self.accuracy = accuracy

    def evaluation(self, preds, actual):
        # self.cm = confusion_matrix(actual, preds)
        accuracy = accuracy_score(actual, preds)

        pt = PrettyTable(['Logistic Regression', 'Accuracy', 'Sensitivity',
                          'Specificity', 'Precision', 'F1 Score'])
        pt.add_row(['Evaluation', accuracy,
                    self.cm[1, 1] / (self.cm[1, 1] + self.cm[1, 0]),
                    self.cm[0, 0] / (self.cm[0, 1] + self.cm[0, 0]),
                    self.cm[1, 1] / (self.cm[1, 1] + self.cm[0, 1]),
                    self.F1_Score])
        print(self.cm, '\n\n', pt)

    # Implementation of neural network


class MultiLayerPerceptron:
    """ Multi Layer Perceptron Model. """
    n_layers: int
    units: List[int]
    biases: List[np.ndarray]
    weights: List[np.ndarray]
    activation: List[Union[None, Callable]]
    activation_derivative: List[Union[None, Callable]]
    loss_functions: List[Callable]
    loss_function_derivatives: List[Callable]

    def __init__(self, units: List[int], activations: List[str], loss_functions: Iterable[str],
                 symmetric_weights: bool = True, seed: int = None) -> None:
        """
            g = activation function
            z = w.T @ a_previous + b
            a = g(z)
        """
        if seed:
            np.random.seed(seed)
        self.units = units
        # logger.info(f"Units per Layer: {self.units}")
        self.n_layers = len(self.units)
        activations = ['linear' if activation_str is None else activation_str
                       for activation_str in activations]
        self.activation = [getattr(self, activation_str)
                           for activation_str in activations]
        self.activation_derivative = [getattr(self, f"{activation_str}_derivative")
                                      for activation_str in activations]
        self.loss_functions = [getattr(self, loss_function) for loss_function in loss_functions]
        self.loss_function_derivatives = [getattr(self, f"{loss_function}_derivative")
                                          for loss_function in loss_functions]
        self.initialize_weights(symmetric_weights)

    def initialize_weights(self, symmetric_weights: bool):
        if symmetric_weights:
            self.biases = [np.random.randn(y, 1) for y in self.units[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(self.units[:-1], self.units[1:])]
        else:
            self.biases = [np.random.rand(y, 1) for y in self.units[1:]]
            self.weights = [np.random.rand(y, x) for x, y in zip(self.units[:-1], self.units[1:])]
        # logger.info(f"Shapes of biases: {[bias.shape for bias in self.biases]}")
        # logger.info(f"Shapes of weights: {[weights.shape for weights in self.weights]}")

    def train(self, data: np.ndarray, one_hot_y: np.ndarray,
              batch_size: int = 1, lr: float = 0.01, momentum: float = 0.0,
              max_epochs: int = 1000, early_stopping: Dict = None,
              shuffle: bool = False, regularization_param: float = 0.0,
              debug: Dict = None, save_data: bool = False,
              min_epoch: int = 1) -> Tuple[List, List, List]:
        # Set Default values
        if not debug:
            debug = {'epochs': 10 ** 10, 'batches': 10 ** 10,
                     'ff': False, 'bp': False, 'w': False, 'metrics': False}
        # Lists to gather accuracies and losses
        accuracies = []
        losses = []
        times = []
        # --- Train Loop --- #
        # data_x, _ = self.x_y_split(data)
        data_x = data
        try:
            for epoch in range(min_epoch, max_epochs + 1):
                if epoch % debug['epochs'] == 0:
                    logger.info(f"Epoch: {epoch}", color="red")
                    show_epoch = True
                else:
                    show_epoch = False
                epoch_timeit = timeit(internal_only=True)
                with epoch_timeit:
                    # Shuffle
                    if shuffle:
                        shuffle_idx = np.random.permutation(data_x.shape[0])
                        data_x = data_x[shuffle_idx, :]
                        one_hot_y = one_hot_y[shuffle_idx, :]
                    # Create Mini-Batches
                    train_batches = [(data_x[k:k + batch_size], one_hot_y[k:k + batch_size])
                                     for k in range(0, data_x.shape[0], batch_size)]
                    # Run mini-batches
                    for batch_ind, (x_batch, one_hot_y_batch) in enumerate(train_batches):
                        batch_ind += 1
                        if show_epoch and batch_ind % debug['batches'] == 0:
                            logger.info(f"  Batch: {batch_ind}", color='yellow')
                        self.run_batch(batch_x=x_batch, batch_y=one_hot_y_batch, lr=lr,
                                       momentum=momentum,
                                       regularization_param=regularization_param, debug=debug)
                        # Calculate Batch Accuracy and Losses
                        if show_epoch and batch_ind % debug['batches'] == 0:
                            accuracy, _ = self.accuracy(data_x, one_hot_y, debug)
                            batch_losses = self.total_loss(data_x, one_hot_y, regularization_param,
                                                           debug)
                            self.print_stats(batch_losses, accuracy, data_x.shape[0], '    ')
                epoch_time = epoch_timeit.total
                # Gather Results
                times.append(epoch_time)
                accuracy, _ = self.accuracy(data_x, one_hot_y, debug)
                epoch_losses = self.total_loss(data_x, one_hot_y, regularization_param, debug)
                accuracies.append(accuracy / data_x.shape[0])
                losses.append(epoch_losses)
                if save_data:
                    self.save_model(epoch, accuracies, losses, times)
                # Calculate Epoch Accuracy and Losses
                if show_epoch:
                    self.print_stats(epoch_losses, accuracy, data_x.shape[0], '  ')
                if early_stopping:
                    if 'max_accuracy' in early_stopping and epoch > early_stopping['wait']:
                        recent_accuracy = accuracies[-1]
                        if recent_accuracy >= early_stopping['max_accuracy']:
                            logger.info(f"Early stopping (Max acc): "
                                        f"{recent_accuracy} = {early_stopping['max_accuracy']}",
                                        color='yellow')
                            break
                    if 'accuracy' in early_stopping and epoch > early_stopping['wait']:
                        recent_accuracy = accuracies[-1] * data_x.shape[0]
                        previous_accuracy = accuracies[-2] * data_x.shape[0]
                        if recent_accuracy - previous_accuracy < early_stopping['accuracy']:
                            logger.info(f"Early stopping (acc): {recent_accuracy}-{previous_accuracy}"
                                        f" = {(recent_accuracy - previous_accuracy)} < "
                                        f"{early_stopping['accuracy']}", color='yellow')
                            break
                    if 'loss' in early_stopping and epoch > early_stopping['wait']:
                        if losses[-1][0][1] - losses[-2][0][1] < early_stopping['loss']:
                            print(losses[-1][0][1], losses[-2][0][1])

                            logger.info(f"Early stopping (loss): "
                                        f"{losses[-1][0][1]:5f}-{losses[-2][0][1]:5f} = "
                                        f"{(losses[-1][0][1] - losses[-2][0][1]):5f} < "
                                        f"{early_stopping['loss']}", color='yellow')
                            break
        except KeyboardInterrupt:
            logger.warn(f"Forcefully stopped after epoch {epoch - 1}")
        if len(accuracies) > 0:
            logger.info(f"Finished after {epoch} epochs", color='red')
            logger.info(f"Avg epoch time: {sum(times) / len(times):.4f} sec(s)", color='yellow')
            logger.info(f"Accumulated epoch time: {sum(times):.4f} sec(s)", color='yellow')
            self.print_stats(epoch_losses, accuracy, data_x.shape[0], '')
            return accuracies, losses, times

    def test(self, data: np.ndarray, one_hot_y: np.ndarray, debug: Dict = None) \
            -> Tuple[float, np.ndarray]:
        if not debug:
            debug = {'epochs': 10 ** 10, 'batches': 10 ** 10,
                     'ff': False, 'bp': False, 'w': False, 'metrics': False}
        # data_x, _ = self.x_y_split(data)
        data_x = data
        accuracy, predictions = self.accuracy(data_x, one_hot_y, debug)
        accuracy /= data_x.shape[0]
        return accuracy, predictions

    @staticmethod
    def print_stats(losses, accuracy, size, padding):
        for loss_type, loss in losses:
            logger.info(f"{padding}{loss_type} Loss: {loss:.5f}")
        logger.info(f"{padding}Accuracy: {accuracy}/{size}")

    def run_batch(self, batch_x: np.ndarray, batch_y: np.ndarray, lr: float,
                  momentum: float, regularization_param: float, debug: Dict):
        for batch_iter, (row_x, row_y) in enumerate(zip(batch_x, batch_y)):
            row_x, row_y = row_x[np.newaxis, :], row_y[:, np.newaxis]
            z, a = self.feed_forward(row_x, debug)
            dw_, db_ = self.back_propagation(row_y, z, a, debug)
            if batch_iter == 0:
                dw = dw_
                db = db_
            else:
                dw = list(map(np.add, dw, dw_))
                db = list(map(np.add, db, db_))

        self.update_weights_and_biases(dw, db, lr, momentum, batch_iter + 1,
                                       regularization_param, debug)

    def feed_forward(self, batch_x: np.ndarray, debug: Dict = None) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        if debug is None:
            debug = {'ff': False}
        z_ = batch_x.T
        z = [z_]
        a_ = z_
        a = [a_]
        for l_ind, layer_units in enumerate(self.units[1:]):
            z_ = self.weights[l_ind] @ a_ + self.biases[l_ind]  # a_ -> a_previous
            z.append(z_)
            a_ = self.activation[l_ind](z_)
            a.append(a_)
            if debug['ff']:
                if l_ind == 0:
                    logger.info("    Feed Forward", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        z{z_.T} = w[{l_ind}]{self.weights[l_ind]} @ a_ + "
                            f"b[{l_ind}]{self.biases[l_ind].T}")
                logger.info(f"        a{a_.T} = g[{l_ind}](z{z_.T})")

        return z, a

    def back_propagation(self, batch_y: np.ndarray, z: List[np.ndarray], a: List[np.ndarray],
                         debug: Dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        db = []
        dw = []
        # Calculate back propagation input which is da of last layer
        da = self.loss_function_derivatives[0](z[-1], a[-1], batch_y)
        for l_ind, layer_units in list(enumerate(self.units))[-1:0:-1]:  # layers: last->2nd
            g_prime = self.activation_derivative[l_ind - 1](z[l_ind])
            try:
                dz = da * g_prime
            except Exception as e:
                print("l_ind: ", l_ind)
                print("layer_units: ", layer_units)
                print("da: ", da)
                print("g_prime: ", g_prime)
                raise e
            db_ = dz
            dw_ = dz @ a[l_ind - 1].T
            da = self.weights[l_ind - 1].T @ dz  # To be used in the next iteration (previous layer)
            db.append(db_)
            dw.append(dw_)
            if debug['bp']:
                if layer_units == self.units[-1]:
                    logger.info("    Back Propagation", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        g_prime{g_prime.shape} = activation_derivative[{l_ind - 1}]"
                            f"(z[{l_ind}]{z[l_ind].shape})"
                            f"{self.activation_derivative[l_ind - 1](z[l_ind]).shape} =\n"
                            f"\t\t\t\t\t\t\t{g_prime.T}")
                logger.info(f"        dz{dz.shape} = da{da.shape} * g_prime{g_prime.shape}")
                logger.info(f"        db{db_.shape} = dz{dz.shape}")
                logger.info(f"        dw = dz{dz.shape} @ a[{l_ind - 1}]{a[l_ind - 1].shape}")
                logger.info(f"        da{da.shape} = self.weights[{l_ind - 1}].T"
                            f"{self.weights[l_ind - 1].T.shape} @ dz{dz.shape} = \n"
                            f"\t\t\t\t\t\t\t{da.T}")

        dw.reverse()
        db.reverse()
        return dw, db

    def update_weights_and_biases(self, dw: List[np.ndarray], db: List[np.ndarray],
                                  lr: float, momentum: float, batch_size: int,
                                  regularization_param: float, debug: Dict) -> None:
        for l_ind, layer_units in enumerate(self.units[:-1]):
            # self.weights[l_ind] -= (lr / batch_size) * dw[l_ind]
            self.weights[l_ind] = (1 - lr * (regularization_param / batch_size)) * self.weights[
                l_ind] - (lr / batch_size) * dw[l_ind] + momentum * self.weights[l_ind]
            self.biases[l_ind] -= (lr / batch_size) * db[l_ind]

            if debug['w']:
                if l_ind == 0:
                    logger.info("    Update Weights", color="cyan")
                logger.info(f"      Layer: {l_ind}, units: {layer_units}", color="magenta")
                logger.info(f"        w({self.weights[l_ind].shape}) -= "
                            f"({lr}/{batch_size}) * dw({dw[l_ind].shape}")
                logger.info(f"        b({self.weights[l_ind].shape}) -= "
                            f"({lr}/{batch_size}) * db({db[l_ind].shape}")

    def save_model(self, epoch, accuracies, losses, times):
        self.save_pickle(var=self, path=f'data/bpnn/model_train_{epoch}.pickle')
        self.save_pickle(var=accuracies, path=f'data/bpnn/accuracies_train_{epoch}.pickle')
        self.save_pickle(var=losses, path=f'data/bpnn/losses_train_{epoch}.pickle')
        self.save_pickle(var=times, path=f'data/bpnn/times_train_{epoch}.pickle')

    @staticmethod
    def save_pickle(var: Any, path: str):
        with open(path, 'wb') as handle:
            pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_model_instance(cls, epoch: int):
        model = cls.load_pickle(f'data/bpnn/model_train_{epoch}.pickle')
        accuracies = cls.load_pickle(f'data/bpnn/accuracies_train_{epoch}.pickle')
        losses = cls.load_pickle(f'data/bpnn/losses_train_{epoch}.pickle')
        times = cls.load_pickle(f'data/bpnn/times_train_{epoch}.pickle')
        return model, accuracies, losses, times

    @staticmethod
    def load_pickle(path: str) -> Any:
        with open(path, 'rb') as handle:
            var = pickle.load(handle)
        return var

    @staticmethod
    def linear(z):
        return z

    linear_derivative = linear

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        z = np.clip(z, -500, 500)  # Handle np.exp overflow
        a = 1.0 / (1.0 + np.exp(-z))
        return a

    @classmethod
    def sigmoid_derivative(cls, a):
        """Derivative of the sigmoid function."""
        return cls.sigmoid(a) * (1 - cls.sigmoid(a))

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z).astype(z.dtype)

    @staticmethod
    def relu_derivative(a):
        return (a > 0).astype(a.dtype)

    @staticmethod
    def tanh(z):
        """ Should use different loss. """
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(a):
        """ Should use different loss. """
        return 1 - a ** 2

    @staticmethod
    def softmax(z):
        # y = np.exp(z - np.max(z))
        # a = y / np.sum(np.exp(z))
        from scipy.special import softmax
        a = softmax(z)
        return a

    softmax_derivative = sigmoid_derivative

    @staticmethod
    def classify(y: np.ndarray) -> np.ndarray:
        total = y.shape[0]
        prediction = np.zeros(total)
        prediction[y.argmax()] = prediction[y.argmax()] = 1
        return prediction

    def predict(self, x: Iterable[np.ndarray], debug: bool = False) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        y_predicted = []
        y_raw_predictions = []
        for x_row in x:
            if debug:
                logger.info(f"  x_row: {x_row[:20].T}", color='white')
            x_row = x_row[np.newaxis, :]
            z, a = self.feed_forward(x_row)
            prediction_raw = a[-1]
            prediction = self.classify(prediction_raw)
            if debug:
                logger.info(f"  prediction_raw: {prediction_raw.T}")
                logger.info(f"  prediction: {prediction}")
            y_raw_predictions.append(prediction_raw)
            y_predicted.append(prediction)
        return y_predicted, y_raw_predictions

    def accuracy(self, data_x: np.ndarray, data_y: np.ndarray,
                 debug: Dict) -> Tuple[int, np.ndarray]:
        if debug['metrics']:
            logger.nl()
            logger.info('Accuracy', color='cyan')
        predictions, _ = self.predict(data_x, debug=debug['metrics'])
        result_accuracy = sum(int(np.array_equal(pred.astype(int), true.astype(int)))
                              for (pred, true) in zip(predictions, data_y))
        if debug['metrics']:
            logger.info(f'result_accuracy: {result_accuracy}')
        return result_accuracy, np.stack(predictions, axis=0)

    def total_loss(self, data_x: np.ndarray, data_y: np.ndarray, regularization_param: float,
                   debug: Dict) -> List[Tuple[str, float]]:
        if debug['metrics']:
            logger.nl()
            logger.info('Total Loss', color='cyan')
        predictions, predictions_raw = self.predict(data_x, debug['metrics'])
        mean_costs = [0.0 for _ in range(len(self.loss_functions))]
        for ind, prediction_raw in enumerate(predictions_raw):
            current_y = data_y[ind]
            for loss_ind, loss_func in enumerate(self.loss_functions):
                mean_costs[loss_ind] += loss_func(prediction_raw, current_y) / len(predictions_raw)
                mean_costs[loss_ind] += 0.5 * (regularization_param / len(predictions_raw)) * sum(
                    np.linalg.norm(w) ** 2
                    for w in self.weights)
            if debug['metrics']:
                logger.info(f'ind: {ind}, prediction_raw: {prediction_raw.T}, current_y: {current_y}')
        costs_with_names = []
        for loss_ind, loss_func in enumerate(self.loss_functions):
            costs_with_names.append((loss_func.__name__, 1.0 / len(data_y) * mean_costs[loss_ind]))
        if debug['metrics']:
            logger.info(f'Mean Costs: {mean_costs}')
        return costs_with_names

    @staticmethod
    def cross_entropy(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a + 1e-15) - (1 - y) * np.log(1 - a + 1e-15)))

    @staticmethod
    def cross_entropy_derivative(z, a, y):
        return a - y

    @staticmethod
    def mse(a, y):
        return np.sum((a - y) ** 2)

    mse_derivative = cross_entropy_derivative

    @staticmethod
    def x_y_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        return dataset[:, :-1], dataset[:, -1][:, np.newaxis].astype(int)

    @staticmethod
    def two_classes_split(dataset: np.ndarray) -> Tuple[np.array, np.array]:
        data_x_c1_idx = dataset[:, -1] == 0
        data_x_c1 = dataset[data_x_c1_idx][:, :-1]
        data_x_c2_idx = dataset[:, -1] == 1
        data_x_c2 = dataset[data_x_c2_idx][:, :-1]
        return data_x_c1, data_x_c2


def train_bpnn(name, dataset, targets, hidden_layers, activations, loss_functions, lr, momentum,
               batch_size, early_stopping, max_epochs, regularization_param, shuffle,
               symmetric_weights, seed, debug, save_data=False):
    logger.nl()
    logger.info(f"Training {name} dataset..")
    # Number of units per layer
    n_units = [int(dataset.shape[1]), *hidden_layers, int(targets.shape[1])]
    logger.info(n_units)
    # Initialize Model
    mlp_model = MultiLayerPerceptron(units=n_units, activations=activations,
                                     symmetric_weights=symmetric_weights,
                                     loss_functions=loss_functions, seed=seed)
    # Train
    accuracies, losses, times = mlp_model.train(data=dataset, one_hot_y=targets,
                                                batch_size=batch_size, lr=lr, momentum=momentum,
                                                shuffle=shuffle, max_epochs=max_epochs,
                                                early_stopping=early_stopping,
                                                regularization_param=regularization_param,
                                                debug=debug, save_data=save_data)

    return mlp_model, accuracies, losses, times


def test_and_plot_bpnn(title, test_set=None, one_hot_targets=None, model=None, accuracies=None,
                       losses=None,
                       times=None,
                       subsample=1, min_acc: float = 0.0, save_predictions: bool = False):
    import types
    # Test the full dataset
    if isinstance(test_set, float):
        test_accuracy = test_set
    elif test_set is None:
        test_accuracy = None
    else:
        model.predict = types.MethodType(MultiLayerPerceptron.predict, model)
        test_accuracy, predictions_onehot = model.test(test_set.copy(), one_hot_targets.copy())
        if save_predictions:
            path = f'data/bpnn'
            path_pred = f'{path}/predicted_y.pickle'
            path_pred_onehot = f'{path}/predicted_onehot_y.pickle'
            predictions = one_hot_unencode(predictions_onehot)
            MultiLayerPerceptron.save_pickle(var=predictions, path=path_pred)
            MultiLayerPerceptron.save_pickle(var=predictions_onehot, path=path_pred_onehot)
    # Plot
    plot_bpnn_results(title=title,
                      test_accuracy=test_accuracy,
                      accuracies=accuracies,
                      losses=losses,
                      times=times,
                      subsample=subsample, min_acc=min_acc)


# Implementation of kmeans clustering algorithm
class kmeans:

    def __init__(self, X_train, max_iter=1000, k=2, dist='euclidean'):
        self.X = X_train
        self.k = k
        self.max_iter = max_iter
        self.centroids = []
        self.switch = []
        self.epoch = []
        self.dist = dist

    def fit(self):
        np.random.seed(42)
        idx = np.random.choice(len(self.X), self.k, replace=False)
        centroids = self.X[idx, :]
        pre_labels = np.argmin(distance.cdist(self.X, centroids, self.dist), axis=1)
        for itr in range(self.max_iter):
            tmp_centroids = []
            for i in range(self.k):

                # handle the case for orphan centroids
                if self.X[pre_labels == i, :].shape[0] == 0:
                    tmp_centroids.append(centroids[i])
                    # print("orphan i ",i)
                else:
                    tmp_centroids.append(self.X[pre_labels == i, :].mean(axis=0))

            # centroids = np.vstack([self.X[pre_labels==i,:].mean(axis=0) for i in range(self.k)])         
            centroids = np.vstack(tmp_centroids)
            current_labels = np.argmin(distance.cdist(self.X, centroids, self.dist), axis=1)
            # print(itr, end=" ")
            # print("swaps ", 100 * ( 1-(sum(pre_labels==current_labels)/len(pre_labels)) ) )

            self.switch.append(100 * (1 - (sum(pre_labels == current_labels) / len(pre_labels))))
            self.epoch.append(itr + 2)
            if np.array_equal(pre_labels, current_labels):
                break
            pre_labels = current_labels
        # print("epochs ",len(self.epoch))
        self.centroids = centroids

    @staticmethod
    def classification_report(y_true, y_pred):
        tn_00 = sum(y_pred[y_true == 0] == y_true[y_true == 0])  # true negatives
        tp_11 = sum(y_pred[y_true == 1] == y_true[y_true == 1])  # true positives
        fp_01 = sum(y_true == 0) - tn_00  # false positives
        fn_10 = sum(y_true == 1) - tp_11  # false negatives
        # confusion_matrix = np.array([[tn_00, fp_01], [fn_10, tp_11]])

        class_0_accuracy = 100.0 * sum(y_pred[y_true == 0] == y_true[y_true == 0]) / sum(y_true == 0)
        class_1_accuracy = 100.0 * sum(y_pred[y_true == 1] == y_true[y_true == 1]) / sum(y_true == 1)

        # print("Kmeans Classification Report:")
        print(f"Overall Accuracy: {round(100.0 * accuracy_score(y_true, y_pred), 2)} %")
        print(f"F1-Score: {round(f1_score(y_true, y_pred), 3)}")
        print(f"F1-Score Macro: {round(f1_score(y_true, y_pred, average='macro'), 3)}")
        print(f"Class 0 accuracy: {round(class_0_accuracy, 2)} %")
        print(f"Class 1 accuracy: {round(class_1_accuracy, 2)} %")

        print("Confusion Matrix:")
        confusion_matrix = PrettyTable(['', 'Predicted 0', 'Predicted 1', 'Total'])
        confusion_matrix.add_row(['Actual 0', tn_00, fp_01, tn_00 + fp_01])
        confusion_matrix.add_row(['Actual 1', fn_10, tp_11, fn_10 + tp_11])
        confusion_matrix.add_row(
            ['Total', tn_00 + fn_10, fp_01 + tp_11, tn_00 + fn_10 + fp_01 + tp_11])
        print(confusion_matrix)

    def predict(self, data, y_true):
        y_pred = np.argmin(distance.cdist(data, self.centroids, 'euclidean'), axis=1)
        if accuracy_score(y_true, y_pred) < 0.5:
            y_pred = 1 - y_pred
        return y_pred

    def plot_membership_switches(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.epoch, self.switch)
        plt.title('Kmeans: Samples Membership Changes vs. Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Membership Changes (%)")
        plt.grid(True)
        plt.show()


# functions used for classification with kNN


def accuracy_score_knn(y, y_model):
    assert len(y) == len(y_model)

    classn = len(np.unique(y))  # number of different classes
    correct_all = y == y_model  # all correctly classified samples

    acc_overall = np.sum(correct_all) / len(y)
    acc_i = []  # list stores classwise accuracy
    for i in np.unique(y):
        acc_i.append(np.sum(correct_all[y == i]) / len(y[y == i]))

    return acc_i, acc_overall


def euclidean(x1, x2):
    edist = np.sqrt(np.sum((x1 - x2) ** 2))
    return edist


def kNN_distances(train, ytrain, test):
    alldist = []
    # Calculate distance between test samples and all samples in training set

    for i in test:  # Loop through all observations in test set

        point_dist = []  # Array to store distances from each observation in test set

        for j in range(len(train)):  # Loop through each point in the training data
            distances = euclidean(np.array(train[j, :]), i)  # Calculate Euclidean distances
            point_dist.append(distances)  # Add distance to array
        point_dist = np.array(point_dist)
        alldist.append(point_dist)
    alldist = np.array(alldist)
    return alldist


def bestk(train, alldist, ytrain, ytest, k_opt):
    accuracy_classwise = []
    accuracy_overall = []

    # Assessing accuracy for different values of k

    for k in k_opt:
        ypredict_knn = kNN(train, alldist, ytrain, ytest, k)
        acc_i, acc_overall = accuracy_score_knn(ytest, ypredict_knn)
        accuracy_overall.append(acc_overall)
        accuracy_classwise.append(acc_i)

    accuracy_overall = np.array(accuracy_overall)  # List of overall accuracy values for each k
    accuracy_classwise = np.array(accuracy_classwise)  # List of classwise accuracy values for each k
    # optimal k for maximizing overall accuracy
    best_k_overall = k_opt[accuracy_overall.argmax()]

    # best overall accuracy
    best_acc_overall = accuracy_overall[accuracy_overall.argmax()]

    # class 0 accuracy for k with best overall accuracy
    class0_acc_overall = accuracy_classwise[accuracy_overall.argmax()][0]

    # class 1 accuracy for k with best overall accuracy
    class1_acc_overall = accuracy_classwise[accuracy_overall.argmax()][1]

    # optimal k for maximizing class 0 accuracy
    best_k_class0 = k_opt[accuracy_classwise[:, 0].argmax()]
    # best class 0 accuracy
    best_acc_class0 = accuracy_classwise[accuracy_classwise[:, 0].argmax()][0]
    # overall accuracy for k with best class 0 accuracy
    overall_acc_class0 = accuracy_overall[accuracy_classwise[:, 0].argmax()]
    # class 1 accuracy for k with best class 0 accuracy
    class1_acc_class0 = accuracy_classwise[accuracy_classwise[:, 0].argmax()][1]
    # optimal k for maximizing class 1 accuracy
    best_k_class1 = k_opt[accuracy_classwise[:, 1].argmax()]
    #  best class 1 accuracy
    best_acc_class1 = accuracy_classwise[accuracy_classwise[:, 1].argmax()][1]
    # overall accuracy for k with best class 1 accuracy
    overall_acc_class1 = accuracy_overall[accuracy_classwise[:, 1].argmax()]
    # class 1 accuracy for k with best class 0 accuracy
    class0_acc_class1 = accuracy_classwise[accuracy_classwise[:, 1].argmax()][0]
    # Combine values for maximizing overall accuracy
    k_overall = [best_k_overall, best_acc_overall, class0_acc_overall, class1_acc_overall]

    # Combine values for maximizing class 0 accuracy
    k_class0 = [best_k_class0, best_acc_class0, overall_acc_class0, class1_acc_class0]

    # Combine values for maximizing class 0 accuracy
    k_class1 = [best_k_class1, best_acc_class1, overall_acc_class1, class0_acc_class1]

    return k_opt, accuracy_overall, accuracy_classwise, k_overall, k_class0, k_class1


def kNN(train, alldist, ytrain, ytest, k):
    ypredict = []

    for i in range(len(alldist)):
        dist = np.argsort(alldist[i])[:k]  # Sort the array of distances and retain k points
        labels = ytrain[dist]  # Getting y-values for k nearest neighbors in training set

        # Sort and use majority voting for different values of k
        lab = np.bincount(labels).argmax()  # Most frequent value in array
        ypredict.append(lab)

    return ypredict


# For evaluation with an sklearn confusion matrix:
def evaluate_cm(sklearn_cm, output):
    accuracy = (sklearn_cm[0, 0] + sklearn_cm[1, 1]) / sklearn_cm.sum()
    precision = sklearn_cm[1, 1] / (sklearn_cm[1, 1] + sklearn_cm[0, 1])
    sensitivity = sklearn_cm[1, 1] / (sklearn_cm[1, 1] + sklearn_cm[1, 0])
    specificity = sklearn_cm[0, 0] / (sklearn_cm[0, 0] + sklearn_cm[0, 1])
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
    if output == 'PRINT':
        print('accuracy: ', accuracy, 'precision: ', precision,
              'sensitivity: ', sensitivity, 'specificity: ',
              specificity, 'f1_score: ', f1_score)

    elif output == 'RETURN':
        return (accuracy, precision, sensitivity, specificity, f1_score)


# Winner-Take-All Code

# Accuracy for WTA
def accuracy_score_wta(y, y_model):
    assert len(y) == len(y_model)

    classn = len(np.unique(y))  # number of different classes
    correct_all = y == y_model  # all correctly classified samples

    acc_overall = np.sum(correct_all) / len(y)
    acc_i = []  # this list7 stores classwise accuracy
    for i in np.unique(y):
        acc_i.append(np.sum(correct_all[y == i]) / len(y[y == i]))
    return acc_i, acc_overall, y, y_model


# Euclidian Distance for WTA
def euclid(x1, x2):
    edist = np.sqrt(np.sum((x1 - x2) ** 2))

    return edist


# WTA Function - Euclidian Distance
def win(Xtest, ytest, k, kcenters, epsilon, max_iter):
    cent_prev = []  # previous center
    group = []
    group_prev = []
    group_change = []
    change = []
    epoch = []

    e = 0

    for it in range(max_iter):

        change = []
        group = []

        for i in Xtest:  # go through all obs in data
            cent_dist = []  # store distances from each obs in test data
            kcenters = np.array(kcenters)

            for j in range(len(kcenters)):  # go through each of kcenters
                distances = euclid((kcenters[j, :]), i)  # distance to each center (min euclidian)
                cent_dist.append(distances)

            label = cent_dist.index(min(cent_dist))  # decision of closest distance
            group.append(label)

            # see lecture 6 slides - moves point closer to closest center

            closest = kcenters[label, :]  # finds closest center & label is the index value

            # w(new) = w(old) + epsilon*(x-w(old)) - equation from lecture slides
            # updating closest cluster
            new_center = closest + epsilon * (i - closest)
            kcenters[label, :] = new_center

        group_prev.append(np.array(group))

        e += 1
        epoch.append(e)
        if it == 0:
            change = 1
            group_change.append(change)
        if it > 0:
            change = (np.sum(np.array(group_prev)[it, :] != np.array(group_prev)[it - 1, :]))
            group_change.append(change / len(Xtest))
        if it > 0 and change == 0.0:
            break
    class_win_acc, overall_win_acc, y, y_model = accuracy_score_wta(ytest, group)
    return epoch, group_change, class_win_acc, overall_win_acc, y, y_model


# choosing initial centers for WTA
import random


def cent(Xtest, k):  # choses random centers to start function

    # random.seed(100)
    nte, nf = Xtest.shape

    ind = np.random.choice(nte, size=k, replace=False)  # index of random rows
    kcenters = Xtest.iloc[ind, :]
    return kcenters, k


# WTA Function - City Block
from scipy.spatial.distance import cityblock


# Creating WTA for different distances -> city block distance
def win_city(Xtest, ytest, k, kcenters, epsilon, max_iter):
    cent_prev = []  # previous center
    group = []
    group_prev = []
    group_change = []
    change = []
    epoch = []

    e = 0

    for it in range(max_iter):

        change = []
        group = []

        for i in Xtest:  # go through all obs in data
            cent_dist = []  # store distances from each obs in test data
            kcenters = np.array(kcenters)

            for j in range(len(kcenters)):  # go through each of kcenters
                distances = cityblock((kcenters[j, :]), i)  # distance to each center (min euclidian)
                cent_dist.append(distances)

            label = cent_dist.index(min(cent_dist))  # decision of closest distance
            group.append(label)

            # see lecture 6 slides - moves point closer to closest center

            closest = kcenters[label, :]  # finds closest center & label is the index value

            # w(new) = w(old) + epsilon*(x-w(old)) - equation from lecture slides
            # updating closest cluster
            new_center = closest + epsilon * (i - closest)
            kcenters[label, :] = new_center

        group_prev.append(np.array(group))

        e += 1
        epoch.append(e)
        if it == 0:
            change = 1
            group_change.append(change)
        if it > 0:
            change = (np.sum(np.array(group_prev)[it, :] != np.array(group_prev)[it - 1, :]))
            group_change.append(change / len(Xtest))
        if it > 0 and change == 0.0:
            break
    class_win_acc, overall_win_acc, y, y_model = accuracy_score(ytest, group)
    return epoch, group_change, class_win_acc, overall_win_acc, y, y_model


# WTA Function - Cosine Similarity
from scipy.spatial.distance import cosine


def win_cosine(Xtest, ytest, k, kcenters, epsilon, max_iter):
    cent_prev = []  # previous center
    group = []
    group_prev = []
    group_change = []
    change = []
    epoch = []

    e = 0

    for it in range(max_iter):

        change = []
        group = []

        for i in Xtest:  # go through all obs in data
            cent_dist = []  # store distances from each obs in test data
            kcenters = np.array(kcenters)

            for j in range(len(kcenters)):  # go through each of kcenters
                distances = cosine((kcenters[j, :]), i)  # distance to each center (min euclidian)
                cent_dist.append(distances)

            label = cent_dist.index(min(cent_dist))  # decision of closest distance
            group.append(label)

            # see lecture 6 slides - moves point closer to closest center

            closest = kcenters[label, :]  # finds closest center & label is the index value

            # w(new) = w(old) + epsilon*(x-w(old)) - equation from lecture slides
            # updating closest cluster
            new_center = closest + epsilon * (i - closest)
            kcenters[label, :] = new_center

        group_prev.append(np.array(group))

        e += 1
        epoch.append(e)
        if it == 0:
            change = 1
            group_change.append(change)
        if it > 0:
            change = (np.sum(np.array(group_prev)[it, :] != np.array(group_prev)[it - 1, :]))
            group_change.append(change / len(Xtest))
        if it > 0 and change == 0.0:
            break
    class_win_acc, overall_win_acc, y, y_model = accuracy_score(ytest, group)
    return epoch, group_change, class_win_acc, overall_win_acc, y, y_model


# WTA Function - Correlation
from scipy.spatial.distance import correlation


def win_corr(Xtest, ytest, k, kcenters, epsilon, max_iter):
    cent_prev = []  # previous center
    group = []
    group_prev = []
    group_change = []
    change = []
    epoch = []

    e = 0

    for it in range(max_iter):

        change = []
        group = []

        for i in Xtest:  # go through all obs in data
            cent_dist = []  # store distances from each obs in test data
            kcenters = np.array(kcenters)

            for j in range(len(kcenters)):  # go through each of kcenters
                distances = correlation((kcenters[j, :]), i)  # distance to each center (min euclidian)
                cent_dist.append(distances)

            label = cent_dist.index(min(cent_dist))  # decision of closest distance
            group.append(label)

            # see lecture 6 slides - moves point closer to closest center

            closest = kcenters[label, :]  # finds closest center & label is the index value

            # w(new) = w(old) + epsilon*(x-w(old)) - equation from lecture slides
            # updating closest cluster
            new_center = closest + epsilon * (i - closest)
            kcenters[label, :] = new_center

        group_prev.append(np.array(group))

        e += 1
        epoch.append(e)
        if it == 0:
            change = 1
            group_change.append(change)
        if it > 0:
            change = (np.sum(np.array(group_prev)[it, :] != np.array(group_prev)[it - 1, :]))
            group_change.append(change / len(Xtest))
        if it > 0 and change == 0.0:
            break
    class_win_acc, overall_win_acc, y, y_model = accuracy_score(ytest, group)
    return epoch, group_change, class_win_acc, overall_win_acc, y, y_model


# WTA Function - Canberra Distance
from scipy.spatial.distance import canberra


def win_can(Xtest, ytest, k, kcenters, epsilon, max_iter):
    cent_prev = []  # previous center
    group = []
    group_prev = []
    group_change = []
    change = []
    epoch = []

    e = 0

    for it in range(max_iter):

        change = []
        group = []

        for i in Xtest:  # go through all obs in data
            cent_dist = []  # store distances from each obs in test data
            kcenters = np.array(kcenters)

            for j in range(len(kcenters)):  # go through each of kcenters
                # Covariance = np.cov(np.transpose(Xtest))
                # inv_covmat = np.linalg.inv(Covariance)
                # V = np.cov(np.array(Xtest.T))
                # IV = np.linalg.inv(V)
                distances = canberra((kcenters[j, :]), i)
                # distances = mahalanobis((kcenters[j, :]), i, inv_covmat) # distance to each center (min euclidian)
                cent_dist.append(distances)

            label = cent_dist.index(min(cent_dist))  # decision of closest distance
            group.append(label)

            # see lecture 6 slides - moves point closer to closest center

            closest = kcenters[label, :]  # finds closest center & label is the index value

            # w(new) = w(old) + epsilon*(x-w(old)) - equation from lecture slides
            # updating closest cluster
            new_center = closest + epsilon * (i - closest)
            kcenters[label, :] = new_center

        group_prev.append(np.array(group))

        e += 1
        epoch.append(e)
        if it == 0:
            change = 1
            group_change.append(change)
        if it > 0:
            change = (np.sum(np.array(group_prev)[it, :] != np.array(group_prev)[it - 1, :]))
            group_change.append(change / len(Xtest))
        if it > 0 and change == 0.0:
            break
    class_win_acc, overall_win_acc, y, y_model = accuracy_score(ytest, group)
    return epoch, group_change, class_win_acc, overall_win_acc, y, y_model


def tune_SKLearn_LR(X_train, X_val, X_test, y_train, y_val, y_test, param):
    grid = {'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'dual': [False],
            'fit_intercept': [True, False],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'C': np.linspace(0.01, 2, 20),
            'multi_class': ['auto', 'ovr', 'multinomial'],
            'warm_start': [True, False]}

    LR = LogisticRegression(max_iter=300)

    if param == 'full':
        print("=========================================")
        print(f"\033[1m Logistic Regression Tuning Results on Full Data\033[0m")
        print("=========================================")

        xtrain, xval, xtest = X_train, X_val, X_test
    elif param == 'pca':

        print("=========================================")
        print(f"\033[1m Logistic Regression Tuning Results on PCA Data\033[0m")
        print("=========================================")

        xtrain, xval, xtest = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)

    start_LR_tune = time()

    LR_random = RandomizedSearchCV(estimator=LR,
                                   param_distributions=grid,
                                   n_iter=100, cv=5,
                                   verbose=0, random_state=44, refit=callable,
                                   n_jobs=-1, scoring=['f1_macro', 'accuracy'])
    # I combined the test and validation data because the RandomizedSearchCV
    # uses cross validation to determine optimal parameters.

    LR_random.fit(pd.concat([xtrain, xval]), pd.concat([y_train, y_val]))

    stop_LR_tune = time()

    time_to_complete = stop_LR_tune - start_LR_tune
    best_param_dict = LR_random.best_params_

    best_model = LogisticRegression(max_iter=300, warm_start=best_param_dict['warm_start'],
                                    solver=best_param_dict['solver'],
                                    penalty=best_param_dict['penalty'],
                                    multi_class=best_param_dict['multi_class'],
                                    fit_intercept=best_param_dict['fit_intercept'],
                                    dual=best_param_dict['dual'],
                                    C=best_param_dict['C']).fit(pd.concat([xtrain, xval]),
                                                                pd.concat([y_train, y_val]))
    preds = best_model.predict(xtest)

    conf_mat_log_reg = confusion_matrix(y_test, preds)
    accuracy_log_reg = accuracy_score(y_test, preds)

    pt = PrettyTable(['Time to Tune (s)', 'Accuracy', 'Sensitivity',
                      'Specificity', 'Precision', 'F1 Score (macro)'])

    pt.add_row([round(time_to_complete, 2), round(accuracy_log_reg, 2),
                round(conf_mat_log_reg[1, 1] / (conf_mat_log_reg[1, 1] + conf_mat_log_reg[1, 0]), 2),
                round(conf_mat_log_reg[0, 0] / (conf_mat_log_reg[0, 1] + conf_mat_log_reg[0, 0]), 2),
                round(conf_mat_log_reg[1, 1] / (conf_mat_log_reg[1, 1] + conf_mat_log_reg[0, 1]), 2),
                round(f1_score(y_test, preds, average='macro'), 2)])
    print('SKL Confusion Matrix: ', conf_mat_log_reg)
    print('LOGISTIC REGRESSION BEST MODEL BASED ON VALIDATION:')
    display(pt)
    print('The best parameters are: ', best_param_dict)

    return (best_model, preds)


def tune_scratch_log_reg(xtrain, ytrain, xval, yval, passes):
    warnings.filterwarnings('ignore')

    # The warning are because exp() is blowing up but since
    # we're dividing by exp() it becomes ~0 which is what we want

    thresh = np.linspace(0.05, 0.95, passes)
    Lrate = np.linspace(0.05, 0.5, passes)
    results, params = [], []

    start_LR_scratch_tune = time()

    for th in thresh:
        for lr in Lrate:
            LR_model = Log_Reg(learning_rate=lr, iters=500)
            LR_model.fit(xtrain, ytrain)
            preds = LR_model.predict(xval, threshold=th)
            LR_model.F1_score_func(yval, preds)
            results.append(LR_model.accuracy)
            params.append([th, lr])

    stop_LR_scratch_tune = time()
    time_LR_scratch_tune = stop_LR_scratch_tune - start_LR_scratch_tune

    opt_n = results.index(max(results))
    print('The optimal threshold and learning rate: ', params[opt_n],
          '\n', 'The highest Accuracy: ',
          results[opt_n], 'Tuning Logistic Regression Scratch took: ', time_LR_scratch_tune, 's')

    opt_params = params[opt_n]

    return opt_params

def perform_xgboost_tuning(X, param):
    # Load the dataset
    X_train, X_val, _, X_train_pca, X_val_pca, \
                            _, y_train, y_val, _ = X
    
    # Find the optimal ratio for scale_pos_weight
    opt_spw = round(y_train.value_counts()[0] / y_train.value_counts()[1], 2)
    parameters = {
                    "n_estimator": [100, 250],
                    "scale_pos_weight": [1, opt_spw, 5],
                    "max_depth": [3, 4, 6, 8],
                    "learning_rate": [1, 0.3, 0.01, 0.05]
                }
                
    if param == 'full':
        print("============================================")
        print(f"\033[1m XGBoost Tuning Results on Full Data\033[0m")
        print("============================================")

        for n in parameters['n_estimator']:
            for spw in parameters['scale_pos_weight']:
                for md in parameters['max_depth']:
                    for lr in parameters['learning_rate']:
                        clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                                        random_state=42, n_estimator=n, scale_pos_weight=spw, \
                                        subsample=0.8, colsample_bytree=0.8, max_depth=md, learning_rate=lr)
                        clf_xg.fit(X_train, y_train)
                        y_pred = clf_xg.predict(X_val)
                        score = accuracy_score(y_val, y_pred)
                        f1_s = f1_score(y_val, y_pred, average='macro')
                        print(f"n_estimator: {n} \t scale_pos_weight: {spw} \t max_depth: {md} \t learning_rate: {lr} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 
    
    elif param == 'pca':
        print("============================================")
        print(f"\033[1m XGBoost Tuning Results on PCA Data\033[0m")
        print("============================================")

        for n in parameters['n_estimator']:
            for spw in parameters['scale_pos_weight']:
                for md in parameters['max_depth']:
                    for lr in parameters['learning_rate']:
                        clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                                        random_state=42, n_estimator=n, scale_pos_weight=spw, \
                                        subsample=0.8, colsample_bytree=0.8, max_depth=md, learning_rate=lr)
                        clf_xg.fit(X_train_pca, y_train)
                        y_pred = clf_xg.predict(X_val_pca)
                        score = accuracy_score(y_val, y_pred)
                        f1_s = f1_score(y_val, y_pred, average='macro')
                        print(f"n_estimator: {n} \t scale_pos_weight: {spw} \t max_depth: {md} \t learning_rate: {lr} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 
    else:
        print("Incorrect argument was passed.")

# Further tune XGBoost on Regularization and Tree Depth
def perform_xgboost_tuning_2(X):
    # Load the dataset
    X_train, X_val, _, X_train_pca, X_val_pca, \
                            _, y_train, y_val, _ = X

    parameters = {
                "max_depth": [12, 20, 16],
                "reg_alpha":[1e-2, 0.1, 0, 10, 100],
                "reg_lambda": [0.1, 1.0, 5.0, 10.0],                    
            }

    print("==========================================================")
    print(f"\033[1m Tune XGBoost on Regularization Parameters (Full Data)\033[0m")
    print("==========================================================")

    for d in parameters['max_depth']:
        for a in parameters['reg_alpha']:
            for l in parameters['reg_lambda']:
                clf_xg = xgb.XGBClassifier(use_label_encoder=False, verbosity = 0, \
                                random_state=42, n_estimator=100, scale_pos_weight=1, \
                                subsample=0.8, colsample_bytree=0.8, max_depth=d, \
                                learning_rate=0.3, reg_alpha=a, reg_lambda=l)
                clf_xg.fit(X_train, y_train)
                y_pred = clf_xg.predict(X_val)
                score = accuracy_score(y_val, y_pred)
                f1_s = f1_score(y_val, y_pred, average='macro')
                print(f"max_depth: {d} \t reg_alpha: {a} \t reg_lambda: {l} \t Accuracy: {round(100*score, 2)}% \t F1 Score(Macro): {round(f1_s, 2)}") 

