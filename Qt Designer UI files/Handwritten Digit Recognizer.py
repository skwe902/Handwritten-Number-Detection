# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Handwritten Digit Recognizer.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(861, 515)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.textConsole = QtWidgets.QTextEdit(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.textConsole.setFont(font)
        self.textConsole.setObjectName("textConsole")
        self.verticalLayout_3.addWidget(self.textConsole)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.totalProgressBar = QtWidgets.QProgressBar(self.groupBox)
        self.totalProgressBar.setProperty("value", 24)
        self.totalProgressBar.setObjectName("totalProgressBar")
        self.verticalLayout_3.addWidget(self.totalProgressBar)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.downloadMNISTBut = QtWidgets.QPushButton(self.groupBox)
        self.downloadMNISTBut.setObjectName("downloadMNISTBut")
        self.horizontalLayout.addWidget(self.downloadMNISTBut)
        self.loadFileBut = QtWidgets.QPushButton(self.groupBox)
        self.loadFileBut.setObjectName("loadFileBut")
        self.horizontalLayout.addWidget(self.loadFileBut)
        self.trainBut = QtWidgets.QPushButton(self.groupBox)
        self.trainBut.setObjectName("trainBut")
        self.horizontalLayout.addWidget(self.trainBut)
        self.cancelBut = QtWidgets.QPushButton(self.groupBox)
        self.cancelBut.setObjectName("cancelBut")
        self.horizontalLayout.addWidget(self.cancelBut)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.textConsole.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:8.25pt;\"><br /></p></body></html>"))
        self.downloadMNISTBut.setText(_translate("MainWindow", "Download MNIST"))
        self.loadFileBut.setText(_translate("MainWindow", "Load"))
        self.trainBut.setText(_translate("MainWindow", "Train"))
        self.cancelBut.setText(_translate("MainWindow", "Cancel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())