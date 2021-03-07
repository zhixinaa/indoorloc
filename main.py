from time import time

from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from qtconsole.qt import QtCore, QtGui

from mywindow import Ui_MainWindow
import sys
import os
import test
import csv


class MainForm(QMainWindow, Ui_MainWindow):
    def select_file(self):
        self.fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd())
        if(self.fileName!= ''):
            self.label.setText(self.fileName)
            self.loadCsv(self.fileName)
            print(self.fileName)
    def getFloor(self):
        t1 = int(round(time() * 1000))
        test.floorPredict(self.fileName)
        t2 = int(round(time() * 1000))-t1
        reply4 = QMessageBox.about(self, "Success!",
                                   "成功处理"+str(self.model.rowCount()) + "条数据，耗时"+str(t2)+"ms")

    def __init__(self):
        self.fileName = ''
        super(MainForm, self).__init__()
        self.setupUi(self)

        self.model = QtGui.QStandardItemModel(self)
        self.tableView.setModel(self.model)
        self.tableView.horizontalHeader().setStretchLastSection(True)

        self.btn_SelectFile.clicked.connect(self.select_file)
        self.btn_StartTest.clicked.connect(self.getFloor)


    def loadCsv(self, fileName):
        self.model.clear()
        with open(fileName, "r") as fileInput:
            for row in csv.reader(fileInput):
                items = [
                    QtGui.QStandardItem(field)
                    for field in row
                ]
                self.model.appendRow(items)





if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainForm()
    win.show()
    sys.exit(app.exec_())
