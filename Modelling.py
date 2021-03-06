import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from numpy.random import seed
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error
import os
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
from matplotlib import pyplot as plt

global X_test, y_test , loc_test
Max_ceng = 13

def data():
    global X_test, y_test ,loc_test
    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    path1 = "./trainingData.csv"
    path2 = "./validationData.csv"
    train = pd.read_csv(path1, index_col=None)
    test = pd.read_csv(path2, index_col=None)
    print('Training dataset (length, width) = {}'.format(str(train.shape)))
    print('Validation dataset (length, width) = {}'.format(str(test.shape)))

    X_train = np.array(train[[x for x in train.columns if 'WAP' in x]])
    lon_train = np.array(train['LONGITUDE'])
    lat_train = np.array(train['LATITUDE'])
    FLOOR = np.array(train['FLOOR'])  # 0~4
    BUILDINGID =  np.array(train['BUILDINGID'])  # 0~2
    loc_train = np.column_stack([lon_train.reshape(-1,1),lat_train.reshape(-1,1)])
    y_train = np.multiply(BUILDINGID, 4) + FLOOR  # 每栋楼4层
    y_train = y_train.reshape(y_train.size, 1)

    X_test = np.array(test[[x for x in test.columns if 'WAP' in x]])
    lon_test = np.array(test['LONGITUDE'])
    lat_test = np.array(test['LATITUDE'])
    FLOORt = np.array(test['FLOOR'])  # 0~4
    BUILDINGIDt = np.array(test['BUILDINGID'])  # 0~2
    y_test = np.multiply(BUILDINGIDt, 4) + FLOORt  # 每栋楼4层
    y_test = y_test.reshape(y_test.size, 1)
    loc_test = np.column_stack([lon_test.reshape(-1,1),lat_test.reshape(-1,1)])


    for i in np.arange(0, X_train.shape[0]):
        for j in np.arange(0, X_train.shape[1]):
            X_train[i][j] = 0 if X_train[i][j] == 100 else 104 + X_train[i][j]

    for i in np.arange(0, X_test.shape[0]):
        for j in np.arange(0, X_test.shape[1]):
            X_test[i][j] = 0 if X_test[i][j] == 100 else 104 + X_test[i][j]
    X_train = X_train / 104
    X_test = X_test / 104
    return X_train, y_train, X_test, y_test ,loc_train


def randomforest(X_train, y_train):
    global X_test, y_test
    # RF
    clf = RandomForestClassifier(n_estimators=70, max_features="auto", max_depth=200, min_samples_split=2,
                                 min_samples_leaf=1, max_leaf_nodes=None)
    clf.fit(X_train, y_train.ravel())
    pickle.dump(clf, open('./model/RF.pkl','wb'))  # 保存

    # 465
    i = j = 0
    lenclf = len(clf.feature_importances_)
    clffeature_importances_ = clf.feature_importances_
    while i < lenclf:
        if (clffeature_importances_[i] == 0):  # 删除
            X_train = np.delete(X_train, j, axis=1)
            X_test = np.delete(X_test, j, axis=1)
            j -= 1
            # print(i)
        i += 1
        j += 1
    print('RF', X_train.shape[1])
    return X_train, y_train


def autoencoder(X_train, X_test):

    encoding_dim = 56
    print('encoding_dim', encoding_dim)
    # this is our input placeholder
    input_img = Input(shape=(X_train.shape[1],))
    act = 'softsign'
    # 编码层
    encoded = Dense(383, activation=act)(input_img)
    encoded = Dense(333, activation=act)(encoded)
    encoded = Dense(256, activation=act)(encoded)
    encoded = Dense(200, activation=act)(encoded)
    encoded = Dense(128, activation=act)(encoded)
    encoded = Dense(96, activation=act)(encoded)
    encoded = Dense(78, activation=act)(encoded)
    encoder_output = Dense(encoding_dim)(encoded)
    # 解码层
    decoded = Dense(78, activation=act)(encoder_output)
    decoded = Dense(96, activation=act)(decoded)
    decoded = Dense(128, activation=act)(decoded)
    decoded = Dense(200, activation=act)(decoded)
    decoded = Dense(256, activation=act)(decoded)
    decoded = Dense(333, activation=act)(decoded)
    decoded = Dense(383, activation=act)(decoded)
    decoded = Dense(X_train.shape[1], activation=act)(decoded)
    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=512, shuffle=True)
    X_test = np.array(encoder.predict(X_test))
    X_train = np.array(encoder.predict(X_train))

    encoder.save('./model/encoder.h5')  # 保存
    return X_train, X_test
def pythagoras(long1, long2, lat1, lat2):
    a = abs(long1-long2)**2
    b = abs(lat1-lat2)**2
    return np.sqrt(a+b)
def knnregress(X_train,loc_train):
    reg_knn = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=2))
    reg_knn.fit(X_train,loc_train)
    loc_preds = reg_knn.predict(X_test)
    pickle.dump(reg_knn, open('./model/reg_knn.pkl', 'wb'))  # 保存
    rsq = r2_score(loc_test, loc_preds)
    print('R-squared score: {:.4f}'.format(rsq))
    print('Mean Squared Error:')
    ll_mse = dict(zip(['Latitude', 'Longitude'],
                      mean_squared_error(loc_test, loc_preds, multioutput='raw_values')))
    for k, v in ll_mse.items():
        print('\t{}: {:.2f}'.format(k, v))

    test_sub = pythagoras(loc_test[:,0], loc_preds[:,0] ,loc_test[:,1], loc_preds[:,1] )

    org_ar = test_sub.mean()
    print('On average, the current model is accurate up to {:.2f}m radius.'.format(org_ar))

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.suptitle('Geographical Distribution of Actual vs. Predicted Records', fontsize=15)
    ax1.scatter(loc_test[:,0],loc_test[:,1], s=10, c='b', marker="s", label='Actual')
    ax1.scatter(loc_preds[:,0],loc_preds[:,1], s=10, c='r', marker="o", label='Predicted')
    plt.xlabel('LATITUDE')
    plt.ylabel('LONGITUDE')
    plt.tight_layout(2)
    plt.legend(loc='upper right')
    plt.show()


def cnnClassifier(X_train1, X_train2,X_test, y_train1):
    input_x, input_y = 8, 7
    X_train1 = X_train1.reshape(len(X_train1), input_x, input_y, 1)
    X_train2 = X_train2.reshape(len(X_train2), input_x, input_y, 1)
    X_test = X_test.reshape(len(X_test), input_x, input_y, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(input_x, input_y, 1), padding='SAME'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='SAME'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(Max_ceng, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train1, y_train1, epochs=50)
    cnn_predict_oneh = model.predict(X_train2)
    CNN_predict = []
    for y in cnn_predict_oneh: CNN_predict.append(y.argmax())
    CNN_predict = np.array(CNN_predict)
    cnn_test_predict_oneh = model.predict(X_test)
    CNN_test_predict = []
    for y in cnn_test_predict_oneh: CNN_test_predict.append(y.argmax())
    CNN_test_predict = np.array(CNN_test_predict)
    test_loss = model.evaluate(X_test, y_test)

    model.save('./model/cnnClassifier.h5')  # 保存
    return CNN_predict, CNN_test_predict


def elmClassifier(X_train1, X_train2, y_train1):
    ex_y_train = np.zeros([y_train1.size, int(np.amax(y_train1) + 1)], dtype=np.float)
    for i in np.arange(0, len(y_train1)):
        ex_y_train[i][int(y_train1[i][0])] = 1.

    elm = ELM(X_train1.shape[1], ex_y_train.shape[1])
    elm.add_neurons(4000, "sigm")
    elm.add_neurons(2060, "rbf_l2")
    elm.train(X_train1, ex_y_train)

    # tarin2
    elm_pred = elm.predict(X_train2)
    ELM_predict = []
    for y in elm_pred:
        ELM_predict.append(y.argmax())
    ELM_predict = np.array(ELM_predict)
    # test
    elm_pred = elm.predict(X_test)
    ELM_test_predict = []
    for y in elm_pred:
        ELM_test_predict.append(y.argmax())
    ELM_test_predict = np.array(ELM_test_predict)
    print('ELM', accuracy_score(y_test, ELM_test_predict))
    elm.save('./model/elmClassifier.pkl')  # 保存
    return ELM_predict, ELM_test_predict


def svmClassifier(X_train1, X_train2, y_train1):
    ##SVM
    kernel = 'poly'
    svm1 = svm.SVC(kernel=kernel, C=1000.0, gamma='scale')
    svm1.fit(X_train1, y_train1.ravel())

    SVM_pre = np.array(svm1.predict(X_train2), dtype=int)
    SVM_test_pre = np.array(svm1.predict(X_test), dtype=int)

    pickle.dump(svm1, open('./model/svmClassifier.pkl','wb'))  # 保存
    print('SVM', kernel, ' accuracy ： ', accuracy_score(y_test, SVM_test_pre))
    return SVM_pre, SVM_test_pre


def xgboostClassifier(X_train1, X_train2, y_train1):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': Max_ceng,  # 类别数，与 multisoftmax 并用
        'gamma': 0.027,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。0.5
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合12
        'lambda': 2.4,  # 0.6 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.56,  # 随机采样训练样本0.56
        'colsample_bytree': 0.4,  # 生成树时进行的列采样
        'min_child_weight': 5,
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.02,  # 如同学习率0.047
        'seed': 1000,
        'nthread': 16,  # cpu 线程数
    }
    plst = list(params.items())
    dtrain = xgb.DMatrix(X_train1, label=y_train1)
    num_rounds = 500
    model = xgb.train(plst, dtrain, num_rounds)

    dtest = xgb.DMatrix(X_train2)
    XGB_predict = np.array(model.predict(dtest), dtype=int)
    dtest = xgb.DMatrix(X_test)
    XGB_test_predict = np.array(model.predict(dtest), dtype=int)

    pickle.dump(model, open('./model/xgboostClassifier.pkl','wb'))  # 保存
    print('XGB', accuracy_score(y_test, XGB_test_predict))
    return XGB_predict, XGB_test_predict


def stacking(CNN_predict, CNN_test_predict,
             ELM_predict, ELM_test_predict,
             SVM_pre, SVM_test_pre,
             XGB_predict, XGB_test_predict):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': Max_ceng,  # 类别数，与 multisoftmax 并用
        'gamma': 0.027,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。0.5
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合12
        'lambda': 2.4,  # 0.6 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.56,  # 随机采样训练样本0.56
        'colsample_bytree': 0.4,  # 生成树时进行的列采样
        'min_child_weight': 5,
        'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.02,  # 如同学习率0.047
        'seed': 1000,
        'nthread': 16,  # cpu 线程数
    }
    # 用前面对train2分类的结果训练最后一层分类器
    x_train_last = np.concatenate([ELM_predict, SVM_pre, XGB_predict, CNN_predict], axis=0)
    x_train_last = x_train_last.reshape(4, int(len(x_train_last) / 4))
    x_train_last = x_train_last.T
    # x_train_last 前5个分类器分别对train2分类的结果
    # X_test_last 前5个分类器分别对测试集分类的结果
    # 最后一层分类器用x_train_last训练，用X_test_last算测试集准确率
    X_test_last = np.concatenate(
        [ELM_test_predict, SVM_test_pre, XGB_test_predict, CNN_test_predict], axis=0)
    X_test_last = X_test_last.reshape(4, int(len(X_test_last) / 4))
    X_test_last = X_test_last.T

    # XGB
    plst = list(params.items())
    dtrain = xgb.DMatrix(x_train_last, label=y_train2)
    num_rounds = 500
    model = xgb.train(plst, dtrain, num_rounds)
    dtest = xgb.DMatrix(X_test_last)
    XGB_predict = model.predict(dtest)
    print('stacking XGB', accuracy_score(y_test, XGB_predict))

    pickle.dump(model, open('./model/stackingClassifier.pkl','wb'))  # 保存




if __name__ == '__main__':
    global X_test, y_test
    X_train, y_train, X_test, y_test ,loc_train= data()
    X_train, y_train = randomforest(X_train, y_train)
    X_train, X_test = autoencoder(X_train, X_test)

    knnregress(X_train, loc_train)

    # train1用来训练第一层分类器，train2训练最后一层分类器
    X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.5)
    X_train2, y_train2 = X_train, y_train
    #X_train1, X_train2, y_train1, y_train2 = X_train, X_train, y_train, y_train

    CNN_predict, CNN_test_predict = cnnClassifier(X_train1, X_train2,X_test, y_train1)
    ELM_predict, ELM_test_predict = elmClassifier(X_train1, X_train2, y_train1)
    SVM_pre, SVM_test_pre = svmClassifier(X_train1, X_train2, y_train1)
    XGB_predict, XGB_test_predict = xgboostClassifier(X_train1, X_train2, y_train1)

    stacking(CNN_predict, CNN_test_predict,
             ELM_predict, ELM_test_predict,
             SVM_pre, SVM_test_pre,
             XGB_predict, XGB_test_predict)
