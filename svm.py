import scipy
import time
import sklearn as skl
from sklearn import *
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = open('./SVHN_official_data/test_32x32.mat','rb')
pics = scipy.io.loadmat(f)
x_test = pics['X']
x_test = np.reshape(x_test, (26032, 32*32*3))
y_test = pics['y']


f = open('./SVHN_official_data/train_32x32.mat','rb')
pics = scipy.io.loadmat(f)
feature = pics['X']
feature = np.reshape(feature, (73257, 32*32*3))
tfm = skl.preprocessing.Normalizer().fit(feature)
feature = tfm.fit_transform(feature)
tfpca = skl.decomposition.PCA(n_components=1024).fit(feature)
feature = tfpca.fit_transform(feature)
label = pics['y']
label = label_binarize(label, classes=list(range(1,11)))
model = OneVsRestClassifier(svm.SVC(gamma='scale'), n_jobs=-1)
print(feature[0], label.shape)
x_train, y_train = feature[:1000], label[:1000]
clf = model.fit(x_train, y_train)
print(clf.score(x_train, y_train))

x_test = tfm.transform(x_test)
x_test = tfpca.transform(x_test)
y_test = label_binarize(y_test, classes=list(range(1,11)))
print(clf.score(x_test, y_test))
print(np.argmax(y_test, axis=1))
print(np.argmax(clf.decision_function(x_test), axis=1).tolist())