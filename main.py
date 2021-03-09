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


class MainForm(QMainWindow, Ui_MainWindow):
    def select_file(self):
        self.fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd())
        if self.fileName != '':
            self.label.setText(self.fileName)
            self.loadDataTableView(self.fileName)
            print(self.fileName)

    def getFloor(self):
        t1 = int(round(time() * 1000))
        floor_predict, y_test = test.floorPredict(self.fileName)

        t2 = int(round(time() * 1000)) - t1
        QMessageBox.about(self, "Success!",
                          "成功处理" + str(self.model.rowCount()) + "条数据，耗时" + str(t2) + "ms")
        self.loadFloorTableView(np.column_stack([
            np.array([i / 4 if i != 12 else 2 for i in y_test]).reshape(-1, 1),  # 楼号
            np.array([i / 4 if i != 12 else 2 for i in floor_predict]).reshape(-1, 1),
            np.array([i % 4 if i != 12 else 4 for i in y_test]).reshape(-1, 1),  # 层号
            np.array([i % 4 if i != 12 else 4 for i in floor_predict]).reshape(-1, 1)]))

    def __init__(self):
        self.fileName = ''
        super(MainForm, self).__init__()
        self.setupUi(self)
        # tableView1
        self.model = QtGui.QStandardItemModel(self)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)
        # tableView2
        self.model_2 = QtGui.QStandardItemModel(self)
        self.tableView_2.setModel(self.model_2)
        # self.tableView_2.horizontalHeader().setStretchLastSection(True)

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
            for i in [0,2]:
                if(items[i].text() == items[i+1].text() ):
                    items[i+1].setBackground(QBrush(QColor(0, 255, 0)))
                else :
                    items[i+1].setBackground(QBrush(QColor(255, 0, 0)))

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
