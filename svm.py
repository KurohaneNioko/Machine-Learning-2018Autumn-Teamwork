import scipy
import pickle
import time
import sklearn as skl
from sklearn import *
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from PIL import Image
import skimage
import os

to_train = 1
to_validate = 0
to_submit = 0
t = 'SVM-030528-0.936'
filename = './log/'+t+'.svm'

f = open('./SVHN_official_data/test_32x32.mat','rb')
pics = scipy.io.loadmat(f)
x_test = pics['X']
x_test = np.transpose(x_test, (3, 0, 1, 2))
x_test = np.array(list(map(lambda x:skimage.color.rgb2grey(x), x_test)))
x_test = np.reshape(x_test, (26032, -1))
y_test = pics['y']

train_max = 73257
train_num = 400
f = open('./SVHN_official_data/train_32x32.mat','rb')
pics = scipy.io.loadmat(f)
feature = pics['X']
feature = np.transpose(feature, (3, 0, 1, 2))[:train_num]
feature = np.array(list(map(lambda x:skimage.color.rgb2grey(x), feature)))
feature = np.reshape(feature, (train_num, -1))
# print(feature[0].tolist())
tfm = skl.preprocessing.Normalizer().fit(feature)
feature = tfm.fit_transform(feature)
# tfpca = skl.decomposition.PCA(n_components=128).fit(feature)
# feature = tfpca.fit_transform(feature)
label = pics['y']
label = label_binarize(label, classes=list(range(1,11)))[:train_num]
x_test = tfm.transform(x_test)
# x_test = tfpca.transform(x_test)
y_test = label_binarize(y_test, classes=list(range(1,11)))
if to_train is not 0:
    model = OneVsRestClassifier(svm.SVC(gamma='scale'), n_jobs=-1)
    x_train, y_train = feature, \
                       label
    clf = model.fit(x_train, y_train)
    v_tru = np.argmax(y_test, axis=1)
    print(v_tru[:30])
    v_pre = np.argmax(clf.decision_function(x_test), axis=1)
    print(v_pre[:30])
    acc = 1 - np.count_nonzero(v_tru-v_pre)/float(len(v_tru))
    print(acc)
    if acc>0.6:
        f = open('./log/SVM-'+time.strftime('%H%M%S')+'-'+str(np.round(acc,3))+'.svm', 'w+b')
        pickle.dump(clf, f)
else:
    svm_folder = './log/'
    m = pickle.load(open(filename, 'rb'))
    if to_validate is not 0:
        v_tru = np.argmax(y_test, axis=1)
        # print(v_tru[:30])
        v_pre = np.argmax(m.decision_function(x_test, ), axis=1)
        # print(v_pre[:30])
        acc = 1 - np.count_nonzero(v_tru - v_pre) / float(len(v_tru))
        print(acc)
    if to_submit is not 0:
        img_list =[]
        for e in range(1800):
            img = np.array(Image.open('./street_data/test/'+str(e)+'.jpg'))
            img = skimage.color.rgb2grey(img)
            img = np.reshape(img, (-1))
            # if e is 0: print(img.tolist())
            img_list.append(img)
        x = tfm.transform(img_list)
        # x = tfpca.transform(x)
        v = np.argmax(m.decision_function(x), axis=1)
        # print(v.tolist())
        T = time.strftime("%H%M%S")
        f = open('./submission_' + T + '.csv', 'w+')
        for e in v:
            f.write(str(e)+'\n')