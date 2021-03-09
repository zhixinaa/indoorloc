from time import time

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
import os
import pandas as pd
global X_test, y_test ,loc_test
Max_ceng = 13

def data(path2):
    #train = pd.read_csv(path1, index_col=None)
    test = pd.read_csv(path2, index_col=None)
    #print('Training dataset (length, width) = {}'.format(str(train.shape)))
    print('Validation dataset (length, width) = {}'.format(str(test.shape)))

    X_test = np.array( test[[x for x in test.columns if 'WAP' in x]])
    lon_test = np.array(test['LONGITUDE'])
    lat_test = np.array(test['LATITUDE'])
    FLOORt = np.array(test['FLOOR'])  # 0~4
    BUILDINGIDt = np.array(test['BUILDINGID'])  # 0~2
    y_test = np.multiply(BUILDINGIDt, 4) + FLOORt  # 每栋楼4层
    y_test = y_test.reshape(y_test.size, 1)
    loc_test = np.column_stack([lon_test.reshape(-1,1),lat_test.reshape(-1,1)])

    for i in np.arange(0, X_test.shape[0]):
        for j in np.arange(0, X_test.shape[1]):
            X_test[i][j] = 0 if X_test[i][j] == 100 else 104 + X_test[i][j]
    X_test = X_test / 104
    #
    # def create_stacked_bar(matrix, axis):
    #     bar_renderers = []
    #     ind = np.arange(matrix.shape[1])
    #     bottoms = np.cumsum(np.vstack((np.zeros(matrix.shape[1]), matrix)), axis=0)[:-1]
    #     for i, row in enumerate(matrix):
    #         r = axis.bar(ind, row, width=0.5, bottom=bottoms[i])
    #         bar_renderers.append(r)
    #     return bar_renderers
    # from matplotlib import pyplot as plt
    #
    # fb_counts = test.groupby(['FLOOR', 'BUILDINGID']).TIMESTAMP.count().reset_index()
    # print( test.groupby(['FLOOR', 'BUILDINGID']))
    # # Inserting dummy rows for missing floor x building combinations
    # for f in np.arange(0, 5):
    #     for b in np.arange(0, 3):
    #         if not ((fb_counts['FLOOR'] == f) & (fb_counts['BUILDINGID'] == b)).any():
    #             fb_counts = fb_counts.append({'FLOOR': f, 'BUILDINGID': b, 'TIMESTAMP': 0}, ignore_index=True)
    # fb_counts.groupby(['BUILDINGID']).TIMESTAMP.sum().reset_index().rename(columns={'TIMESTAMP': 'RECORDS'})
    #
    # pivot_fb = fb_counts.pivot(index='FLOOR', columns='BUILDINGID')
    # print('pivot_fb.values', pivot_fb.values)
    # buildings = list(set(fb_counts['BUILDINGID'].tolist()))
    # print('buildings',buildings)
    # plt.figure(figsize=(15, 5))
    # bars = create_stacked_bar(pivot_fb.values, plt)
    # # Plot formatting
    # plt.legend((reversed([x[0] for x in bars])), (4, 3, 2, 1, 0), fancybox=True)
    # plt.title('Number of Records by Building and Floor', fontsize=20)
    # plt.xticks(buildings)
    # plt.xlabel('Buildings')
    # plt.ylabel('Number of Records')
    #
    # plt.show()
    return X_test, y_test  ,loc_test


def randomforest(X_test):
    # RF
    clf = pickle.load(open('./model/RF.pkl','rb'))  # 加载
    Temp = []
    # 465
    i = 0
    clfLen = len(clf.feature_importances_)
    clf_feature_importances_ = clf.feature_importances_
    while i < clfLen:
        if (clf_feature_importances_[i] != 0):
            Temp.append(i)
        i += 1
    print('RF', len(Temp))
    return X_test[:,Temp]


def autoencoder(X_test):
    encoding_dim = 56
    print('encoding_dim', encoding_dim)
    encoder = tf.keras.models.load_model('./model/encoder.h5')
    X_test = np.array(encoder.predict(X_test))
    return X_test

def knnregress(X_test):
    from matplotlib import pyplot as plt
    reg_knn = pickle.load( open('./model/reg_knn.pkl', 'rb'))  # 保存
    loc_preds = reg_knn.predict(X_test)
    return loc_preds

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
    floor_predict = model.predict(dtest)
    accuracy = accuracy_score(y_test, floor_predict)
    print('stacking XGB', accuracy)
    return  floor_predict


def floorPredict(path2):
    global X_test, y_test , loc_test
    t1 = int(round(time() * 1000))

    X_test, y_test, loc_test = data(path2)
    print('tdata',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))
    print(X_test.shape)

    X_test = randomforest(X_test)
    print('tRF',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))
    print(X_test.shape)

    X_test = autoencoder(X_test)
    print('tENCODER',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))
    print(X_test.shape)

    loc_preds = knnregress(X_test)
    print('tKNN', int(round(time() * 1000)) - t1)
    t1 = int(round(time() * 1000))

    CNN_test_predict = cnnClassifier(X_test)
    print('tCNN',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))


    ELM_test_predict = elmClassifier(X_test)
    print('tELM',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))

    SVM_test_pre = svmClassifier(X_test)
    print('tSVM',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))

    XGB_test_predict = xgboostClassifier(X_test)
    print('tXGB',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))

    Stacking_test_predict = stacking(CNN_test_predict,
             ELM_test_predict,
             SVM_test_pre,
             XGB_test_predict)

    print('tSTACKING',int(round(time() * 1000))-t1)
    t1 = int(round(time() * 1000))
    return Stacking_test_predict,y_test , loc_preds ,loc_test



if __name__ == '__main__':
    path2 = "./validationData.csv"
    floorPredict(path2)
