import numpy as np
from time import time
from typing import *
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from prettytable import PrettyTable
from project_libs import ColorizedLogger
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report


logger = ColorizedLogger('Models', 'green')

np.seterr(divide='raise')

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
        first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                     np.linalg.inv(self.first_and_second_case_cov))
        first_term = -(1 / 2) * np.matmul(first_term_dot_1,
                                          (self.x_test[sample] - self.means[n_class]))
        second_term = np.log(priors[n_class])
        g = first_term + second_term
        return g

    def _build_g_quadratic(self, sample, n_class, priors: List[float]):
        current_covs = self.covs[n_class]
        try:
            first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                         np.linalg.inv(current_covs))
        except np.linalg.LinAlgError as e:
            logger.debug(f"{e}")
            current_covs += + 10e-5
            first_term_dot_1 = np.matmul((self.x_test[sample] - self.means[n_class]).T,
                                         np.linalg.inv(current_covs))
        except Exception as e:
            logger.debug(f"{e}")
            first_term_dot_1 = (self.x_test[sample] - self.means[n_class]).T / current_covs

        first_term = -(1 / 2) * np.matmul(first_term_dot_1,
                                          (self.x_test[sample] - self.means[n_class]))
        try:
            second_term = -(1 / 2) * np.log(np.linalg.det(current_covs))
        except Exception as e:
            logger.debug(f"{e}")
            second_term = -(1 / 2) * np.log(current_covs)
        third_term = np.log(priors[n_class])
        g = first_term + second_term + third_term
        return g

    def predict(self, mtype: str, priors: List[float] = None,
                first_and_second_case_cov_type: str = 'avg') -> np.ndarray:
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


# Implementation of kmeans clustering algorithm
class kmeans:
    
    def __init__(self, X_train, max_iter=1000, k=2, dist='euclidean'):
        self.X = X_train
        self.k = k
        self.max_iter=max_iter
        self.centroids = []
        self.switch = []
        self.epoch = []
        self.dist = dist
    
    def fit(self):
        np.random.seed(42)
        idx = np.random.choice(len(self.X), self.k, replace=False)
        centroids = self.X[idx, :]
        pre_labels = np.argmin(distance.cdist(self.X, centroids, self.dist),axis=1)
        for itr in range(self.max_iter):
            tmp_centroids = []
            for i in range(self.k):
                
                # handle the case for orphan centroids
                if self.X[pre_labels==i,:].shape[0] == 0:
                    tmp_centroids.append(centroids[i])
                    # print("orphan i ",i)
                else:
                    tmp_centroids.append(self.X[pre_labels==i,:].mean(axis=0))
                    
            # centroids = np.vstack([self.X[pre_labels==i,:].mean(axis=0) for i in range(self.k)])         
            centroids = np.vstack(tmp_centroids)         
            current_labels = np.argmin(distance.cdist(self.X, centroids, self.dist),axis=1)
            # print(itr, end=" ")
            # print("swaps ", 100 * ( 1-(sum(pre_labels==current_labels)/len(pre_labels)) ) )
            
            self.switch.append( 100 * ( 1-(sum(pre_labels==current_labels)/len(pre_labels)) ) )
            self.epoch.append(itr+2)
            if np.array_equal(pre_labels, current_labels):
                break
            pre_labels = current_labels
        # print("epochs ",len(self.epoch))
        self.centroids = centroids

    @staticmethod
    def classification_report(y_true, y_pred):
        tn_00 = sum(y_pred[y_true==0]==y_true[y_true==0]) # true negatives
        tp_11 = sum(y_pred[y_true==1]==y_true[y_true==1]) # true positives
        fp_01 = sum(y_true==0) - tn_00 # false positives
        fn_10 = sum(y_true==1) - tp_11 # false negatives
        # confusion_matrix = np.array([[tn_00, fp_01], [fn_10, tp_11]])

        class_0_accuracy=100.0*sum(y_pred[y_true==0]==y_true[y_true==0])/sum(y_true==0)
        class_1_accuracy=100.0*sum(y_pred[y_true==1]==y_true[y_true==1])/sum(y_true==1)
        
        # print("Kmeans Classification Report:")
        print(f"Overall Accuracy: {round(100.0*accuracy_score(y_pred, y_true), 2)} %") 
        print(f"F1-Score: {round(f1_score(y_true, y_pred), 3)}")
        print(f"Class 0 accuracy: {round(class_0_accuracy, 2)} %" ) 
        print(f"Class 1 accuracy: {round(class_1_accuracy, 2)} %") 
        
        print("Confusion Matrix:")
        confusion_matrix = PrettyTable(['','Predicted 0', 'Predicted 1','Total'])
        confusion_matrix.add_row(['Actual 0', tn_00, fp_01, tn_00+fp_01])
        confusion_matrix.add_row(['Actual 1', fn_10, tp_11, fn_10+tp_11])
        confusion_matrix.add_row(['Total', tn_00+fn_10, fp_01+tp_11, tn_00+fn_10+fp_01+tp_11])        
        print(confusion_matrix)

    def predict(self, data, y_true):      
        y_pred = np.argmin(distance.cdist(data, self.centroids, 'euclidean'),axis=1)
        if accuracy_score(y_pred, y_true) < 0.5:
            y_pred = 1-y_pred
        return y_pred

    def plot_membership_switches(self):
        plt.figure(figsize=(10, 8))   
        plt.plot(self.epoch, self.switch)
        plt.title('Kmeans: Samples Membership Changes vs. Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Membership Changes (%)")
        plt.grid(True)
        plt.show()