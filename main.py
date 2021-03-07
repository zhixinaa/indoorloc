from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
import sklearn as sk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import svm
from numpy.random import seed

seed(1)
from tensorflow import random

random.set_seed(2)
import random as rn

rn.seed(3)
from hpelm import ELM
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

path1 = "./trainingData.csv"
path2 = "./validationData.csv"
X_train = np.array(np.loadtxt(path1, skiprows=1, delimiter=",", usecols=np.arange(0, 520), dtype=int))
FLOOR = np.array(np.loadtxt(path1, skiprows=1, delimiter=",", usecols=(522,), dtype=int))  # 0~4
BUILDINGID = np.array(np.loadtxt(path1, skiprows=1, delimiter=",", usecols=(523,), dtype=int))  # 0~2
y_train = np.multiply(BUILDINGID, 4) + FLOOR  # 每栋楼4层
y_train = y_train.reshape(y_train.size, 1)
Max_ceng = np.amax(y_train) + 1
X_test = np.array(np.loadtxt(path2, skiprows=1, delimiter=",", usecols=np.arange(0, 520), dtype=int))
FLOORt = np.array(np.loadtxt(path2, skiprows=1, delimiter=",", usecols=(522,), dtype=int))  # 0~4
BUILDINGIDt = np.array(np.loadtxt(path2, skiprows=1, delimiter=",", usecols=(523,), dtype=int))  # 0~2
y_test = np.multiply(BUILDINGIDt, 4) + FLOORt  # 每栋楼4层
y_test = y_test.reshape(y_test.size, 1)
for i in np.arange(0, X_train.shape[0]):
    for j in np.arange(0, X_train.shape[1]):
        X_train[i][j] = 0 if X_train[i][j] == 100 else 104 + X_train[i][j]

for i in np.arange(0, X_test.shape[0]):
    for j in np.arange(0, X_test.shape[1]):
        X_test[i][j] = 0 if X_test[i][j] == 100 else 104 + X_test[i][j]
X_train = X_train / 104
X_test = X_test / 104

# RF
clf = RandomForestClassifier(n_estimators=70, max_features="auto", max_depth=200, min_samples_split=2,
                             min_samples_leaf=1, max_leaf_nodes=None)
clf.fit(X_train, y_train.ravel())
print(clf.feature_importances_)
temp = np.sort(clf.feature_importances_)
# 465
i = j = 0
while i < len(clf.feature_importances_):
    if (clf.feature_importances_[i] == 0):  # 删除
        X_train = np.delete(X_train, j, axis=1)
        X_test = np.delete(X_test, j, axis=1)
        j -= 1
        # print(i)
    i += 1
    j += 1
print('RF',X_train.shape[1])

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

# 一部分用来训练前面几个分类器，另一部分训练最后一层分类器
#X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.7)
#X_train2, y_train2 = X_train, y_train
X_train1, X_train2, y_train1, y_train2 = X_train, X_train, y_train, y_train
input_x ,input_y= 8,7
X_train = X_train.reshape(len(X_train), input_x ,input_y, 1)
X_train1 = X_train1.reshape(len(X_train1),input_x ,input_y, 1)
X_train2 = X_train2.reshape(len(X_train2), input_x ,input_y, 1)
X_test = X_test.reshape(len(X_test), input_x ,input_y, 1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=(input_x ,input_y, 1), padding='SAME'),
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

X_train = X_train.reshape(len(X_train), 56)
X_train1 = X_train1.reshape(len(X_train1), 56)
X_train2 = X_train2.reshape(len(X_train2), 56)
X_test = X_test.reshape(len(X_test), 56)

ex_y_train = np.zeros([y_train1.size, int(np.amax(y_train1) + 1)], dtype=np.float)
for i in np.arange(0, len(y_train1)):
    ex_y_train[i][int(y_train1[i][0])] = 1.

elm = ELM(X_train1.shape[1], ex_y_train.shape[1])
elm.add_neurons(4000, "sigm")
elm.add_neurons(2060, "rbf_l2")
elm.train(X_train1, ex_y_train)
elm_pred = elm.predict(X_test)
ELM_predict = []
for y in elm_pred:
    ELM_predict.append(y.argmax())
accuracy = accuracy_score(y_test, ELM_predict)
print('ELM', accuracy)
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
##SVM_poly
kernel = 'poly'
classifier1 = svm.SVC(kernel=kernel, C=1000.0, gamma='scale')
classifier1.fit(X_train1, y_train1.ravel())
SVM_poly_pre = np.array(classifier1.predict(X_test), dtype=int)
accuracy = accuracy_score(y_test, SVM_poly_pre)
print('SVM', kernel, ' accuracy ： ', accuracy)
SVM_poly_pre = np.array(classifier1.predict(X_train2), dtype=int)
SVM_poly_test_pre = np.array(classifier1.predict(X_test), dtype=int)


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
dtrain = xgb.DMatrix(X_train, label=y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)
dtest = xgb.DMatrix(X_test)
XGB_predict = model.predict(dtest)
accuracy = accuracy_score(y_test, XGB_predict)
print('XGB', accuracy)
dtest = xgb.DMatrix(X_train2)
XGB_predict = np.array(model.predict(dtest), dtype=int)
dtest = xgb.DMatrix(X_test)
XGB_test_predict = np.array(model.predict(dtest), dtype=int)

# 用前面对train2分类的结果训练最后一层分类器
x_train_last = np.concatenate([ELM_predict, SVM_poly_pre, XGB_predict, CNN_predict], axis=0)
x_train_last = x_train_last.reshape(4, int(len(x_train_last) / 4))
x_train_last = x_train_last.T
# x_train_last 前5个分类器分别对train2分类的结果
# X_test_last 前5个分类器分别对测试集分类的结果
# 最后一层分类器用x_train_last训练，用X_test_last算测试集准确率
X_test_last = np.concatenate(
    [ELM_test_predict, SVM_poly_test_pre, XGB_test_predict, CNN_test_predict], axis=0)
X_test_last = X_test_last.reshape(4, int(len(X_test_last) / 4))
X_test_last = X_test_last.T

# XGB
plst = list(params.items())
dtrain = xgb.DMatrix(x_train_last, label=y_train2)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)
dtest = xgb.DMatrix(X_test_last)
XGB_predict = model.predict(dtest)
accuracy = accuracy_score(y_test, XGB_predict)
print('XGB', accuracy)
