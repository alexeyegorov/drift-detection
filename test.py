# coding=utf-8
"""
This is a test script for the three algorithms (Adwin, DDM, Stream Volatility)
implemented in this project. The script takes the input dataset of Power Supply.
The test is based on Prequential Evaluation, and monitors 3 indicators of
performance: Accuracy, Time and memory usage
"""

# Authors: Wenjun Bai <vivianbai.cn@gmail.com>
#          Shu Shang <ignatius.sun@gmail.com>
#          Duyen Phuc Nguyen <nguyenduyenphuc@gmail.com>
# License: BSD 3 clause

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.naive_bayes import GaussianNB
from classifiers.detector_classifier import DetectorClassifier
from drift_detector.adwin import Adwin
from drift_detector.DDM import DDM
from drift_detector.stream_volatility.volatility_detector import VolatilityDetector
from evluation.metrics import Exact_match
from evluation.prequential import prequential_evaluation, get_errors

np.random.seed(0)

print('Load data')

"""
Dataset: Power Supply Dataset

Download
--------
moa.cms.waikato.ac.nz/datasets/
http://www.cse.fau.edu/ ∼ xqzhu/stream.html

Data Structure
--------------
|"date"|"day"|"period"|"nswprice"|"nswdemand"|"vicprice"|"vicdemand"|"transfer"|"class"|

shape = (45312, 9)
n_features = 8
label = column['class'] = {"UP", "DOWN"}
"""
df = pd.read_csv("data/elecNormNew.csv")
df['class'] = df['class'].map({'UP': 0, 'DOWN': 1})
L = 8
N_train = 1000

labels = df.columns.values.tolist()[L:]
data = df.values
T = len(data)
Y = data[:, L:]
X = data[:, 0:L]

print("Experimentation")

h = [DetectorClassifier(GaussianNB(), Adwin()),
     DetectorClassifier(GaussianNB(), VolatilityDetector(drift_detector=Adwin(), size=32)),
     DetectorClassifier(GaussianNB(), DDM()),
     GaussianNB()]
E_pred, E_time, E_usage = prequential_evaluation(X, Y, h, N_train)

print("Evaluation")

E = np.zeros((len(h), T - N_train))
for m in range(len(h)):
    E[m] = get_errors(Y[N_train:], E_pred[m], J=Exact_match)

print("Plot Results")
print("---------------------------------------")
w = 200
fig, axes = plt.subplots(nrows=3, ncols=1)
fig.tight_layout()
for m in range(len(h)):
    acc = np.mean(E[m, :])
    time = np.mean(E_time[m, :])
    usage = np.mean(E_usage[m, :])
    if h[m].__class__.__name__ == 'DetectorClassifier':
        print(h[m].__class__.__name__)
        print(h[m].get_detector_name())
    else:
        print(h[m].__class__.__name__)
    print("Exact Match %3.2f" % np.mean(acc))
    # print("Running Time  %3.2f" % np.mean(time))
    if h[m].__class__.__name__ == 'DetectorClassifier':
        print("Number of detected drifts: %d" % h[m].num_change_detected)
    print("---------------------------------------")
    acc_run = np.convolve(E[m, :], np.ones((w,)) / w, 'same')
    acc_time = np.convolve(E_time[m, :], np.ones((w,)) / w, 'same')
    acc_usage = np.convolve(E_usage[m, :], np.ones((w,)) / w, 'same')
    if h[m].__class__.__name__ == 'DetectorClassifier':
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(acc_run)), acc_run, '-', label=h[m].get_detector_name())
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(len(acc_time)), acc_time, '-', label=h[m].get_detector_name())
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(len(acc_usage)), acc_usage, '-', label=h[m].get_detector_name())
    else:
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(len(acc_run)), acc_run, '-', label=h[m].__class__.__name__)
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(len(acc_time)), acc_time, '-', label=h[m].__class__.__name__)
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(len(acc_usage)), acc_usage, '-', label=h[m].__class__.__name__)

plt.subplot(3, 1, 1)
plt.xlabel('Instance(samples)')
plt.ylabel('Accuracy(exact match)')
plt.title('Performance(acc)')
plt.legend(loc='best')
plt.subplot(3, 1, 2)
plt.xlabel('Instance(samples)')
plt.ylabel('Running time(ms)')
plt.title('Performance(Running time)')
plt.legend(loc='best')
plt.subplot(3, 1, 3)
plt.xlabel('Instance(samples)')
plt.ylabel('Memory usage (%MEM)')
plt.title('Performance(Memory usage)')
plt.legend(loc='best')
plt.show()
