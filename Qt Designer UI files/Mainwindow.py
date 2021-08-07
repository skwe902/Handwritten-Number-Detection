# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1267, 986)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.clearBut = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clearBut.sizePolicy().hasHeightForWidth())
        self.clearBut.setSizePolicy(sizePolicy)
        self.clearBut.setMinimumSize(QtCore.QSize(200, 0))
        self.clearBut.setMaximumSize(QtCore.QSize(300, 16777215))
        self.clearBut.setObjectName("clearBut")
        self.verticalLayout_2.addWidget(self.clearBut, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.randomBut = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.randomBut.sizePolicy().hasHeightForWidth())
        self.randomBut.setSizePolicy(sizePolicy)
        self.randomBut.setMinimumSize(QtCore.QSize(200, 0))
        self.randomBut.setMaximumSize(QtCore.QSize(300, 16777215))
        self.randomBut.setObjectName("randomBut")
        self.verticalLayout_2.addWidget(self.randomBut, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.modelBut = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.modelBut.sizePolicy().hasHeightForWidth())
        self.modelBut.setSizePolicy(sizePolicy)
        self.modelBut.setMinimumSize(QtCore.QSize(200, 0))
        self.modelBut.setMaximumSize(QtCore.QSize(300, 16777215))
        self.modelBut.setObjectName("modelBut")
        self.verticalLayout_2.addWidget(self.modelBut, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.recognizeBut = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.recognizeBut.sizePolicy().hasHeightForWidth())
        self.recognizeBut.setSizePolicy(sizePolicy)
        self.recognizeBut.setMinimumSize(QtCore.QSize(200, 0))
        self.recognizeBut.setMaximumSize(QtCore.QSize(500, 16777215))
        self.recognizeBut.setObjectName("recognizeBut")
        self.verticalLayout_2.addWidget(self.recognizeBut, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 2)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.DrawnDigitDisplay = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DrawnDigitDisplay.setFont(font)
        self.DrawnDigitDisplay.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.DrawnDigitDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.DrawnDigitDisplay.setObjectName("DrawnDigitDisplay")
        self.verticalLayout.addWidget(self.DrawnDigitDisplay, 0, QtCore.Qt.AlignBottom)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(200, 200))
        self.label.setMaximumSize(QtCore.QSize(200, 200))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../Gitrepo/project-1-team_01/CroppedDrawnDigit.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.probabillityHeaderText = QtWidgets.QLabel(self.centralwidget)
        self.probabillityHeaderText.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.probabillityHeaderText.setFont(font)
        self.probabillityHeaderText.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.probabillityHeaderText.setAutoFillBackground(False)
        self.probabillityHeaderText.setObjectName("probabillityHeaderText")
        self.verticalLayout_3.addWidget(self.probabillityHeaderText, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setProperty("value", 24)
        self.progressBar_2.setObjectName("progressBar_2")
        self.verticalLayout_3.addWidget(self.progressBar_2, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3.setProperty("value", 24)
        self.progressBar_3.setObjectName("progressBar_3")
        self.verticalLayout_3.addWidget(self.progressBar_3, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_4.setProperty("value", 24)
        self.progressBar_4.setObjectName("progressBar_4")
        self.verticalLayout_3.addWidget(self.progressBar_4, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_5 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_5.setProperty("value", 24)
        self.progressBar_5.setObjectName("progressBar_5")
        self.verticalLayout_3.addWidget(self.progressBar_5, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_6 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_6.setProperty("value", 24)
        self.progressBar_6.setObjectName("progressBar_6")
        self.verticalLayout_3.addWidget(self.progressBar_6, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setProperty("value", 24)
        self.progressBar_7.setObjectName("progressBar_7")
        self.verticalLayout_3.addWidget(self.progressBar_7, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_8 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_8.setProperty("value", 24)
        self.progressBar_8.setObjectName("progressBar_8")
        self.verticalLayout_3.addWidget(self.progressBar_8, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_9 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_9.setProperty("value", 24)
        self.progressBar_9.setObjectName("progressBar_9")
        self.verticalLayout_3.addWidget(self.progressBar_9, 0, QtCore.Qt.AlignVCenter)
        self.mostLikelyNumText = QtWidgets.QLabel(self.centralwidget)
        self.mostLikelyNumText.setMaximumSize(QtCore.QSize(300, 30))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.mostLikelyNumText.setFont(font)
        self.mostLikelyNumText.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.mostLikelyNumText.setObjectName("mostLikelyNumText")
        self.verticalLayout_3.addWidget(self.mostLikelyNumText, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout.setStretch(0, 3)
        self.verticalLayout.setStretch(3, 7)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 8)
        self.horizontalLayout.setStretch(1, 2)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1267, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionetc = QtWidgets.QAction(MainWindow)
        self.actionetc.setObjectName("actionetc")
        self.menuFile.addAction(self.actionetc)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.clearBut.setText(_translate("MainWindow", "Clear"))
        self.randomBut.setText(_translate("MainWindow", "Random"))
        self.modelBut.setText(_translate("MainWindow", "Model"))
        self.recognizeBut.setText(_translate("MainWindow", "Recognize"))
        self.DrawnDigitDisplay.setText(_translate("MainWindow", "Image Drawn"))
        self.probabillityHeaderText.setText(_translate("MainWindow", "Class Probabillity"))
        self.mostLikelyNumText.setText(_translate("MainWindow", "insert num here"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionetc.setText(_translate("MainWindow", "etc"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())