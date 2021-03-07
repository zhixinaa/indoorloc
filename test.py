import numpy as np
import tensorflow as tf
import xgboost as xgb
from numpy.random import seed
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
seed(1)
from tensorflow import random

random.set_seed(2)
import random as rn

rn.seed(3)
from hpelm import ELM
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
from sklearn.ensemble import RandomForestClassifier
import pickle

global X_test, y_test
Max_ceng = 13

def data():
    path2 = "./validationData.csv"
    X_test = np.array(np.loadtxt(path2, skiprows=1, delimiter=",", usecols=np.arange(0, 520), dtype=int))
    FLOORt = np.array(np.loadtxt(path2, skiprows=1, delimiter=",", usecols=(522,), dtype=int))  # 0~4
    BUILDINGIDt = np.array(np.loadtxt(path2, skiprows=1, delimiter=",", usecols=(523,), dtype=int))  # 0~2
    y_test = np.multiply(BUILDINGIDt, 4) + FLOORt  # 每栋楼4层
    y_test = y_test.reshape(y_test.size, 1)

    for i in np.arange(0, X_test.shape[0]):
        for j in np.arange(0, X_test.shape[1]):
            X_test[i][j] = 0 if X_test[i][j] == 100 else 104 + X_test[i][j]
    X_test = X_test / 104
    return X_test, y_test


def randomforest(X_test):
    # RF
    clf = pickle.load(open('./model/RF.pkl','rb'))  # 加载

    # 465
    i = j = 0
    while i < len(clf.feature_importances_):
        if (clf.feature_importances_[i] == 0):  # 删除
            X_test = np.delete(X_test, j, axis=1)
            j -= 1
            # print(i)
        i += 1
        j += 1
    print('RF', X_test.shape[1])
    return X_test


def autoencoder(X_test):
    encoding_dim = 56
    print('encoding_dim', encoding_dim)
    encoder = tf.keras.models.load_model('./model/encoder.h5')
    X_test = np.array(encoder.predict(X_test))
    return X_test


def cnnClassifier(X_test):
    input_x, input_y = 8, 7
    X_test = X_test.reshape(len(X_test), input_x, input_y, 1)

    model = tf.keras.models.load_model('./model/cnnClassifier.h5')

    cnn_test_predict_oneh = model.predict(X_test)
    CNN_test_predict = []
    for y in cnn_test_predict_oneh: CNN_test_predict.append(y.argmax())
    CNN_test_predict = np.array(CNN_test_predict)

    accuracy = accuracy_score(y_test, CNN_test_predict)
    print('CNN', accuracy)

    return CNN_test_predict


def elmClassifier(X_test):
    elm = ELM(X_test.shape[1], y_test.shape[1])
    elm.load('./model/elmClassifier.pkl')
    # test
    elm_pred = elm.predict(X_test)
    ELM_test_predict = []
    for y in elm_pred:
        ELM_test_predict.append(y.argmax())
    ELM_test_predict = np.array(ELM_test_predict)
    accuracy = accuracy_score(y_test, ELM_test_predict)
    print('ELM', accuracy)
    return ELM_test_predict


def svmClassifier(X_test):
    ##SVM
    svm1 = pickle.load(open('./model/svmClassifier.pkl','rb'))  # 保存
    SVM_test_pre = np.array(svm1.predict(X_test), dtype=int)
    print('SVM', ' accuracy ： ', accuracy_score(y_test, SVM_test_pre))
    return SVM_test_pre


def xgboostClassifier(X_test):

    model =pickle.load( open('./model/xgboostClassifier.pkl','rb'))

    ##
    dtest = xgb.DMatrix(X_test)
    XGB_test_predict = np.array(model.predict(dtest), dtype=int)
    accuracy = accuracy_score(y_test, XGB_test_predict)
    print('XGB', accuracy)
    return XGB_test_predict


def stacking( CNN_test_predict,
              ELM_test_predict,
              SVM_test_pre,
              XGB_test_predict):

    X_test_last = np.concatenate(
        [ELM_test_predict, SVM_test_pre, XGB_test_predict, CNN_test_predict], axis=0)
    X_test_last = X_test_last.reshape(4, int(len(X_test_last) / 4))
    X_test_last = X_test_last.T

    # XGB

    model = pickle.load( open('./model/stackingClassifier.pkl','rb'))
    dtest = xgb.DMatrix(X_test_last)
    XGB_predict = model.predict(dtest)
    accuracy = accuracy_score(y_test, XGB_predict)
    print('stacking XGB', accuracy)


if __name__ == '__main__':
    global X_test, y_test
    X_test, y_test = data()
    X_test = randomforest(X_test)
    X_test = autoencoder(X_test)

    CNN_test_predict = cnnClassifier(X_test)
    ELM_test_predict = elmClassifier(X_test)
    SVM_test_pre = svmClassifier(X_test)
    XGB_test_predict = xgboostClassifier(X_test)

    stacking( CNN_test_predict,
              ELM_test_predict,
              SVM_test_pre,
              XGB_test_predict)
