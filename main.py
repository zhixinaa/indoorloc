from time import time

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from PyQt5 import QtCore, QtGui

from mywindow import Ui_MainWindow
import sys
import os
import test
import csv
import numpy as np
import math

import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# 创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)
        self.sizeHint()

    # 第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    def plotcos(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)

    def plot_position(self, loc_preds, loc_test):
        ax1 = self.axes
        self.fig.suptitle('Geographical Distribution of Actual vs. Predicted Records', fontsize=15)
        ax1.scatter(loc_test[:, 0], loc_test[:, 1], s=10, c='b', marker="s", label='Actual')
        ax1.scatter(loc_preds[:, 0], loc_preds[:, 1], s=10, c='r', marker="o", label='Predicted')
        ax1.set_xlabel('LATITUDE', )
        ax1.set_ylabel('LONGITUDE')
        # self.fig.tight_layout(None,3.0)
        ax1.legend(loc='upper right')

    def plot_floor(self, buildiing_floor):
        def create_stacked_bar(matrix, axis):
            bar_renderers = []
            ind = np.arange(matrix.shape[1])
            bottoms = np.cumsum(np.vstack((np.zeros(matrix.shape[1]), matrix)), axis=0)[:-1]
            for i, row in enumerate(matrix):
                r = axis.bar(ind, row, width=0.5, bottom=bottoms[i])
                bar_renderers.append(r)
            return bar_renderers

        pivot_fb = np.zeros([5, 3], int)
        for i in buildiing_floor:
            if (i[0] != i[1] or i[2] != i[3]):
                pivot_fb[int(i[3])][int(i[0])] += 1
        buildings = [0, 1, 2]
        print('pivot_fb', pivot_fb)
        ax1 = self.axes
        bars = create_stacked_bar(pivot_fb, ax1)
        # Plot formatting
        ax1.legend((reversed([x[0] for x in bars])), (4, 3, 2, 1, 0), fancybox=True)
        ax1.set_title('Number of Records by Building and Floor', fontsize=20)
        ax1.set_xticks(buildings)
        ax1.set_xlabel('Buildings')
        ax1.set_ylabel('Number of Records')


class MainForm(QMainWindow, Ui_MainWindow):
    def select_file(self):
        self.fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd())
        if self.fileName != '':
            self.label.setText(self.fileName)
            self.loadDataTableView(self.fileName)
            print(self.fileName)

    def getFloor(self):
        t1 = int(round(time() * 1000))
        floor_predict, y_test, loc_preds, loc_test = test.floorPredict(self.fileName)
        print(floor_predict)
        print(y_test)
        print(loc_preds)
        t2 = int(round(time() * 1000)) - t1
        QMessageBox.about(self, "Success!",
                          "成功处理" + str(self.model.rowCount()) + "条数据，耗时" + str(t2) + "ms")
        buildiing_floor = np.column_stack([
            np.array([i / 4 if i != 12 else 2 for i in y_test]).reshape(-1, 1),  # 楼号
            np.array([i / 4 if i != 12 else 2 for i in floor_predict]).reshape(-1, 1),
            np.array([i % 4 if i != 12 else 4 for i in y_test]).reshape(-1, 1),  # 层号
            np.array([i % 4 if i != 12 else 4 for i in floor_predict]).reshape(-1, 1)])
        self.loadFloorTableView(buildiing_floor)

        # 第五步：定义MyFigure类的一个实例
        self.F = MyFigure(width=3, height=2, dpi=100)
        self.F.plot_position(loc_preds, loc_test)
        # 第六步：在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F, 1, 0, )
        # self.gridlayout.addWidget(self.tableView, 0, 0 )

        self.F2 = MyFigure(width=4, height=2, dpi=80)
        self.F2.plot_floor(buildiing_floor)
        self.F3 = MyFigure(width=4, height=2, dpi=80)
        self.F3.plot_position(loc_preds, loc_test)
        self.vboxLayout_1 = QVBoxLayout()
        self.vboxLayout_1.addWidget(self.F2)
        self.vboxLayout_1.addWidget(self.F3)
        self.gridlayout_2.addLayout(self.vboxLayout_1, 0, 1)

    def __init__(self):
        self.fileName = ''
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.F = None
        self.F2 = None
        self.F3 = None
        self.gridlayout = None
        self.vboxLayout_1 = None
        # tableView1
        self.model = QtGui.QStandardItemModel(self)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        # tableView2
        self.tableView_2 = QtWidgets.QTableView(self)
        self.model_2 = QtGui.QStandardItemModel(self)
        # self.tableView_2.horizontalHeader().setSectionResizeMode()
        self.tableView_2.setModel(self.model_2)
        self.tableView_2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.gridlayout_2 = QGridLayout(self.groupBox_2)  # 继承容器groupBox
        self.gridlayout_2.addWidget(self.tableView_2, 0, 0)

        # connect
        self.btn_SelectFile.clicked.connect(self.select_file)
        self.btn_StartTest.clicked.connect(self.getFloor)

    def loadDataTableView(self, fileName):
        self.model.clear()
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)

    def loadFloorTableView(self, array):
        self.model_2.clear()

        for row in array:
            items = [
                QtGui.QStandardItem(str(int(field)))
                for field in row
            ]
            # 根据是否预测正确，修改单元格颜色
            for i in [0, 2]:
                if (items[i].text() == items[i + 1].text()):
                    items[i + 1].setBackground(QBrush(QColor(0, 255, 0)))
                else:
                    items[i + 1].setBackground(QBrush(QColor(255, 0, 0)))

            self.model_2.appendRow(items)
        self.model_2.setHeaderData(0, QtCore.Qt.Horizontal, "实际楼号")
        self.model_2.setHeaderData(1, QtCore.Qt.Horizontal, "预测楼号")
        self.model_2.setHeaderData(2, QtCore.Qt.Horizontal, "实际层号")
        self.model_2.setHeaderData(3, QtCore.Qt.Horizontal, "预测层号")
        self.tabWidget.setCurrentIndex(1)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
