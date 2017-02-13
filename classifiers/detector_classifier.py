""" Detector Classifier, a wrapper to combine the classifier and drift detector  """

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

import numpy as np

from sklearn import clone
from sklearn.metrics import accuracy_score


class DetectorClassifier():
    """
    A detector classifier is a classifier combined with a drift detector.
    This class serves as wrapper to combine a classifier and a drift detector together.
    """
    def __init__(self, clf, drift_detector):
        """
        Initialize a detector classifier.

        Parameters
        ----------
        clf: a classifier, like Naive Bayes classifier
        drift_detector: a drift detector, like adwin, DDM
        """
        self.classes = None
        self.clf = clf
        self.drift_detector = drift_detector
        self.num_change_detected = 0

    def fit(self, X, y):
        """Fit drift detector classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            return self
        """
        self.clf.fit(X, y)
        self.classes = np.unique(y)
        return self

    def partial_fit(self, X, y):
        """Incremental fit on a batch of samples.
        This method is expected to be called several times consecutively
        on different chunks of a dataset so as to implement out-of-core
        or online learning.

        This is especially useful when the whole dataset is too big to fit in
        memory at once.

        This method has some performance and numerical stability overhead,
        hence it is better to call partial_fit on chunks of data that are
        as large as possible (as long as fitting in the memory budget) to
        hide the overhead.

        Parameters
        ----------
        X : array-like, shape(n_samples, n_features)
            Training vectors, when n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape(n_samples, )
            Target values.

        Returns
        -------
        self : object
            return self.
        """
        pre_y = self.clf.predict(X)
        if self.drift_detector.set_input(accuracy_score(pre_y, y)):
            self.num_change_detected += 1
            self.clf = clone(self.clf)
            # print("change detected...")
            # self.clf.fit(X, y)
            self.clf.partial_fit(X, y, classes=self.classes)
        else:
            self.clf.partial_fit(X, y)
        return self

    def predict(self, X):
        """
        Perform prediction on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        array, shape = [n_samples]
            Predicted target values for X
        """
        return self.clf.predict(X)

    def get_detector_name(self):
        return self.drift_detector.__class__.__name__
