from __future__ import print_function
import torch
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F
import time
from threading import Thread
from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
from PIL import Image, ImageFilter
import glob
from PIL import ImageOps
import matplotlib.pyplot as plt
import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QWidget, QPushButton, QVBoxLayout, QLabel, QDesktopWidget, QFileDialog
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2
import math
from scipy import ndimage

import random
import AnotherWindow
import TrainingImages
import TestingImages

#Author: Vishnu Hu and Andy Kwon. for COMPSYS 302 at the University of Auckland

device = 'cuda' if cuda.is_available() else 'cpu'

class DefaultWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        DefaultWindow.pointerSelf = self
        self.initUI()
        #set title of our app
        self.setWindowTitle('Handwritten Digit Recognizer') 
        #set icon
        self.setWindowIcon(QIcon(os.getcwd() + r'\App Icon.ico'))
        #Resize
        self.setGeometry(300, 300, 800, 800)
        #the lines below centre the program window

        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

        ##skeleton code for the main window starts FROM HERE
        self.centralwidget = QtWidgets.QWidget(self)
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

        #initialzie clear button
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
        self.clearBut.clicked.connect(self.clearButPressed)

        #initialize random button
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
        self.randomBut.clicked.connect(self.randomButPressed)

        #initialize model button
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
        self.modelBut.clicked.connect(self.modelButtonPressed)

        #initialize recognize button
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
        self.recognizeBut.clicked.connect(self.recognizeButPressed)

        self.verticalLayout_2.setStretch(0, 2)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 2)
        self.verticalLayout_2.setStretch(3, 2)
        self.verticalLayout.addLayout(self.verticalLayout_2)

        #initialize label that Displays drawn digit
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
        self.label.setPixmap(QtGui.QPixmap(os.getcwd() + r"\CroppedDrawnDigit.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label, 0, QtCore.Qt.AlignHCenter)
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

        ##Initialize PROGRESS BARS
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setAlignment(Qt.AlignCenter)
        self.progressBar.setFormat('0')
        self.verticalLayout_3.addWidget(self.progressBar, 0, QtCore.Qt.AlignVCenter)

        self.progressBar_1 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_1.setProperty("value", 0)
        self.progressBar_1.setObjectName("progressBar_1")
        self.progressBar_1.setAlignment(Qt.AlignCenter)
        self.progressBar_1.setFormat('1')
        self.verticalLayout_3.addWidget(self.progressBar_1, 0, QtCore.Qt.AlignVCenter)


        self.progressBar_2 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.progressBar_2.setAlignment(Qt.AlignCenter)
        self.progressBar_2.setFormat('2')

        self.verticalLayout_3.addWidget(self.progressBar_2, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_3 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")
        self.progressBar_3.setAlignment(Qt.AlignCenter)
        self.progressBar_3.setFormat('3')

        self.verticalLayout_3.addWidget(self.progressBar_3, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_4 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")
        self.progressBar_4.setAlignment(Qt.AlignCenter)
        self.progressBar_4.setFormat('4')

        self.verticalLayout_3.addWidget(self.progressBar_4, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_5 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_5.setProperty("value", 0)
        self.progressBar_5.setObjectName("progressBar_5")
        self.progressBar_5.setAlignment(Qt.AlignCenter)
        self.progressBar_5.setFormat('5')

        self.verticalLayout_3.addWidget(self.progressBar_5, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_6 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_6.setProperty("value", 0)
        self.progressBar_6.setObjectName("progressBar_6")
        self.progressBar_6.setAlignment(Qt.AlignCenter)
        self.progressBar_6.setFormat('6')

        self.verticalLayout_3.addWidget(self.progressBar_6, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_7 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_7.setProperty("value", 0)
        self.progressBar_7.setObjectName("progressBar_7")
        self.progressBar_7.setAlignment(Qt.AlignCenter)
        self.progressBar_7.setFormat('7')

        self.verticalLayout_3.addWidget(self.progressBar_7, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_8 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_8.setProperty("value", 0)
        self.progressBar_8.setObjectName("progressBar_8")
        self.progressBar_8.setAlignment(Qt.AlignCenter)
        self.progressBar_8.setFormat('8')

        self.verticalLayout_3.addWidget(self.progressBar_8, 0, QtCore.Qt.AlignVCenter)
        self.progressBar_9 = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar_9.setProperty("value", 0)
        self.progressBar_9.setObjectName("progressBar_9")
        self.progressBar_9.setAlignment(Qt.AlignCenter)
        self.progressBar_9.setFormat('9')
        self.verticalLayout_3.addWidget(self.progressBar_9, 0, QtCore.Qt.AlignVCenter)

        #Initialize the label holding most likely value
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
        self.setCentralWidget(self.centralwidget)


        #MAKING Painter (covers whole application GUI)

        #decleare Qimage
        self.image = QImage(self.size(), QImage.Format_RGB32)
        # making image color to white
        self.image.fill(Qt.white)
        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 43
        # default color
        self.brushColor = Qt.black
        # QPoint object to tract the point
        self.lastPoint = QPoint()



        #this retranslate command just runs the initialization of the buttons
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)
        self.firstInitializeFlag = True
        #show the body
        self.show()

    def retranslateUi(self, MainWindow): #this function names all the buttons
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Handwritten Digit Recognizer", "Handwritten Digit Recognizer"))
        self.clearBut.setText(_translate("Handwritten Digit Recognizer", "Clear"))
        self.randomBut.setText(_translate("Handwritten Digit Recognizer", "Random"))
        self.modelBut.setText(_translate("Handwritten Digit Recognizer", "About Model"))
        self.recognizeBut.setText(_translate("Handwritten Digit Recognizer", "Recognize"))
        self.DrawnDigitDisplay.setText(_translate("Handwritten Digit Recognizer", "Image Drawn"))
        self.probabillityHeaderText.setText(_translate("Handwritten Digit Recognizer", "Class Probabillity"))
        self.mostLikelyNumText.setText(_translate("Handwritten Digit Recognizer", ""))


    def initUI(self): #this initialzes the Menubar
        ##actions 
        exitAction = QAction('Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        trainAction = QAction('Train Model', self)
        trainAction.setStatusTip('Train Model')
        trainAction.triggered.connect(lambda: self.trainModel())

        viewTrain = QAction('View Training Images', self)
        viewTrain.setStatusTip('View the Training Images')
        viewTrain.triggered.connect(lambda: self.viewTrainingImagesPressed())

        viewTest = QAction('View Testing Images', self)
        viewTest.setStatusTip('View the Testing Images')
        viewTest.triggered.connect(lambda: self.viewTestingImagesPressed())
        #buttons

        self.statusBar()

        menubar = self.menuBar()
        filemenu = menubar.addMenu('&File')
        filemenu.addAction(trainAction)
        filemenu.addAction(exitAction)

        filemenu = menubar.addMenu('&View')
        filemenu.addAction(viewTrain)
        filemenu.addAction(viewTest)

    def mousePressEvent(self, event):
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):
        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.image)
            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize, 
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())
            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()
  
    # method for mouse left button release
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False
            
    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)
        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def resizeEvent(self, event): #this needs to be here otherwise painter coord map will be confused when we resize window
        #reinitialize the painter coord map
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

    def clearButPressed(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

    def randomButPressed(self): #showes a random image from the dataset 
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

        testset = datasets.MNIST('mnist_data_test', download=False, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)

        dataiterTest = iter(testloader) # creating a iterator
        imagesTest, labelsTest = dataiterTest.next()

        randomNumber = random.randint(0,10000)
        for i in range(0,1):
            plt.subplot(1, 1, i+1)
            plt.axis('off')
            plt.imshow(imagesTest[randomNumber].numpy().squeeze(), cmap='gray_r')
            plt.savefig("Random.png")

        image=Image.open("Random.png")
        image.load()
        image = image.convert("RGB") 
        invertedImage = ImageOps.invert(image)
        invertedImage = invertedImage.resize((28,28))
        invertedImage.save(os.getcwd() + r"\Random.png")


        self.label.setPixmap(QtGui.QPixmap(os.getcwd() + r"\Random.png"))
        self.update()

        image = Image.open("Random.png").convert('RGB') #from: https://stackoverflow.com/questions/51803437/how-do-i-use-a-saved-model-in-pytorch-to-predict-the-label-of-a-never-before-see 
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
        inputImage = transform(imagesTest[randomNumber])  #this converts the .png image into a tensor which can be loaded into the dataset
        inputImage = inputImage.unsqueeze(1) #for cnn remove if its something else
        inputImage = inputImage.to(device)
        trainedModel = torch.load('trainedModel.pth', map_location=torch.device(device))
        trainedModel.eval()
        with torch.no_grad():
            output = trainedModel(inputImage)

        ps = torch.exp(output)
        probab = list(ps.cpu().numpy()[0])
        print("Predicted Digit =", probab.index(max(probab))) #https://github.com/amitrajitbose/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_CPU.ipynb
        print(probab) 
        ##send the values to pgBar
        self.progressBar.setValue(int(100*probab[0]))
        self.progressBar_1.setValue(int(100*probab[1]))
        self.progressBar_2.setValue(int(100*probab[2]))
        self.progressBar_3.setValue(int(100*probab[3]))
        self.progressBar_4.setValue(int(100*probab[4]))
        self.progressBar_5.setValue(int(100*probab[5]))
        self.progressBar_6.setValue(int(100*probab[6]))
        self.progressBar_7.setValue(int(100*probab[7]))
        self.progressBar_8.setValue(int(100*probab[8]))
        self.progressBar_9.setValue(int(100*probab[9]))

        for i in range(0,10):
            if probab[i]<0.01:
                probab[i] = 0 # to remove any probability less than 1%

        self.progressBar.setFormat('0 probability: {0} %'.format(round(probab[0]*100)))
        self.progressBar_1.setFormat('1 probability: {0} %'.format(round(probab[1]*100)))
        self.progressBar_2.setFormat('2 probability: {0} %'.format(round(probab[2]*100)))
        self.progressBar_3.setFormat('3 probability: {0} %'.format(round(probab[3]*100)))
        self.progressBar_4.setFormat('4 probability: {0} %'.format(round(probab[4]*100)))
        self.progressBar_5.setFormat('5 probability: {0} %'.format(round(probab[5]*100)))
        self.progressBar_6.setFormat('6 probability: {0} %'.format(round(probab[6]*100)))
        self.progressBar_7.setFormat('7 probability: {0} %'.format(round(probab[7]*100)))
        self.progressBar_8.setFormat('8 probability: {0} %'.format(round(probab[8]*100)))
        self.progressBar_9.setFormat('9 probability: {0} %'.format(round(probab[9]*100)))

        #display mostlikely num to user
        self.mostLikelyNumText.setText(QtCore.QCoreApplication.translate("MainWindow", "Most Likely Value: {}".format(probab.index(max(probab)))))

    def modelButtonPressed(self):
       QMessageBox.question(self, 'Model Description', 'This model is a deep learning neural network as it utilizes two fully connected layers \n\nThe model uses three 2D Convolutional layers and ReLU as activation function. \n\nTo avoid overfitting, the model features 4 dropouts and has been trained with 50 epoch and 16 batch size. \n\nTo understand more about the model, open "Net3.py"', QMessageBox.Ok, QMessageBox.Ok)

    def recognizeButPressed(self):

        x = os.getcwd() + r"\DrawnDigit.png"
        self.image.save(x)

        filePath = os.getcwd() + r"\DrawnDigit.png"

        image=Image.open(filePath)
        image.load()

        image = image.convert("RGB") 
        invertedImage = ImageOps.invert(image)
        imageBox = invertedImage.getbbox() # remove black border https://stackoverflow.com/questions/9870876/getbbox-method-from-python-image-library-pil-not-working

        if imageBox is not None: #i need this here because it gives error when you insert blank image
            ##set the box around the number (20 pixes out in each direction BEFORE RESIZE)
            imageBox = imageBox[0], imageBox[1], imageBox[2], imageBox[3] #https://www.geeksforgeeks.org/python-pil-imagepath-path-getbbox-method/
            cropped = image.crop(imageBox)
            if (imageBox[2] - imageBox[0]) > (imageBox[3] - imageBox[1]):
                cropped = ImageOps.expand(cropped, border = int((imageBox[2] - imageBox[0]) * 0.4), fill = 'white') #this keeps the ratio of border 8 : image 20
            else:
                cropped = ImageOps.expand(cropped, border = int((imageBox[3] - imageBox[1]) * 0.4), fill = 'white')

            #removes the whitespace
            #Resizes to 28x28 pixels png image
            #save the image
        else:
            cropped = Image.new('L', [28,28], 255)

        cropped = ImageOps.invert(cropped)
        cropped = cropped.resize((28,28))

        cropped.save(os.getcwd() + r"\CroppedDrawnDigit.png")

        self.label.setPixmap(QtGui.QPixmap(os.getcwd() + r"\CroppedDrawnDigit.png"))

        self.image.fill(Qt.white)
        self.update()

        #DefaultWindow.imageprepare("CroppedDrawnDigit.png")
        image = Image.open("CroppedDrawnDigit.png").convert('RGB') #from: https://stackoverflow.com/questions/51803437/how-do-i-use-a-saved-model-in-pytorch-to-predict-the-label-of-a-never-before-see 
        transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
        inputImage = transform(image)  #this converts the .png image into a tensor which can be loaded into the dataset
        inputImage = inputImage.unsqueeze(1) #for cnn remove if its something else
        inputImage = inputImage.to(device)
        trainedModel = torch.load('trainedModel.pth', map_location=torch.device(device))
        trainedModel.eval()
        with torch.no_grad():
            output = trainedModel(inputImage)

        ps = torch.exp(output)
        probab = list(ps.cpu().numpy()[0])
        print("Predicted Digit =", probab.index(max(probab))) #https://github.com/amitrajitbose/handwritten-digit-recognition/blob/master/handwritten_digit_recognition_CPU.ipynb
        print(probab) 
        ##send the values to pgBar
        self.progressBar.setValue(int(100*probab[0]))
        self.progressBar_1.setValue(int(100*probab[1]))
        self.progressBar_2.setValue(int(100*probab[2]))
        self.progressBar_3.setValue(int(100*probab[3]))
        self.progressBar_4.setValue(int(100*probab[4]))
        self.progressBar_5.setValue(int(100*probab[5]))
        self.progressBar_6.setValue(int(100*probab[6]))
        self.progressBar_7.setValue(int(100*probab[7]))
        self.progressBar_8.setValue(int(100*probab[8]))
        self.progressBar_9.setValue(int(100*probab[9]))

        for i in range(0,10):
            if probab[i]<0.01:
                probab[i] = 0 # to remove any probability less than 1%

        self.progressBar.setFormat('0 probability: {0} %'.format(round(probab[0]*100)))
        self.progressBar_1.setFormat('1 probability: {0} %'.format(round(probab[1]*100)))
        self.progressBar_2.setFormat('2 probability: {0} %'.format(round(probab[2]*100)))
        self.progressBar_3.setFormat('3 probability: {0} %'.format(round(probab[3]*100)))
        self.progressBar_4.setFormat('4 probability: {0} %'.format(round(probab[4]*100)))
        self.progressBar_5.setFormat('5 probability: {0} %'.format(round(probab[5]*100)))
        self.progressBar_6.setFormat('6 probability: {0} %'.format(round(probab[6]*100)))
        self.progressBar_7.setFormat('7 probability: {0} %'.format(round(probab[7]*100)))
        self.progressBar_8.setFormat('8 probability: {0} %'.format(round(probab[8]*100)))
        self.progressBar_9.setFormat('9 probability: {0} %'.format(round(probab[9]*100)))

        #display mostlikely num to user
        self.mostLikelyNumText.setText(QtCore.QCoreApplication.translate("MainWindow", "Most Likely Value: {}".format(probab.index(max(probab)))))

    def viewTrainingImagesPressed(self):
        self.newWindow = TrainingImages.TrainingImages()
        self.newWindow.show()
    
    def viewTestingImagesPressed(self):
        self.newWindow = TestingImages.TestingImages()
        self.newWindow.show()
    
    def trainModel(self):
        if(self.firstInitializeFlag):
            self.w = AnotherWindow.AnotherWindow()
            self.firstInitializeFlag = False
        print('') #clear the PGbar 
        self.w.show()
    
    def chooseModel(self):
        QMessageBox.about(self, "Model 1", "Load Complete!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DefaultWindow()
    sys.exit(app.exec_())
