import torch
import torchvision
from torchvision import datasets, transforms

from PyQt5.QtGui import * 
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import *

import matplotlib.pyplot as plt

import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np


class TestingImages(QMainWindow):
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

    testset = datasets.MNIST('mnist_data_train', download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=True)

    dataiterTest = iter(testloader) # creating a iterator
    imagesTest, labelsTest = dataiterTest.next()

    num_of_images = 100
    for index in range(1, num_of_images+1):
        plt.subplot(10, 10, index)
        plt.axis('off')
        plt.imshow(imagesTest[index].numpy().squeeze(), cmap='gray_r')
    plt.savefig("initTest.png") #creates an initial image to display 

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Testing Images")
        self.centralwidget = QWidget()
        self.setCentralWidget(self.centralwidget)
        self.lay = QVBoxLayout(self.centralwidget)
        
        self.label = QLabel(self)
        self.pixmap = QPixmap('initTest.png')
        self.label.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(), self.pixmap.height())
        
        self.lay.addWidget(self.label)

        self.button = QPushButton('Show Previous', self)
        self.button.setToolTip('This is a previous button')
        self.button.move(20,450)
        self.button.resize(100,30)
        self.button.clicked.connect(self.previousClicked)

        self.button = QPushButton('Show Next', self)
        self.button.setToolTip('This is a next button')
        self.button.move(530,450)
        self.button.resize(100,30)
        self.button.clicked.connect(self.nextClicked)

        self.checkBox1 = QCheckBox("1", self)
        self.checkBox1.stateChanged.connect(self.uncheck)
        self.checkBox1.move(15,20)
        self.checkBox1.resize(320,40)

        self.checkBox2 = QCheckBox("2", self)
        self.checkBox2.stateChanged.connect(self.uncheck)
        self.checkBox2.move(15,40)
        self.checkBox2.resize(320,40)

        self.checkBox3 = QCheckBox("3", self)
        self.checkBox3.stateChanged.connect(self.uncheck)
        self.checkBox3.move(15,60)
        self.checkBox3.resize(320,40)

        self.checkBox4 = QCheckBox("4", self)
        self.checkBox4.stateChanged.connect(self.uncheck)
        self.checkBox4.move(15,80)
        self.checkBox4.resize(320,40)

        self.checkBox5 = QCheckBox("5", self)
        self.checkBox5.stateChanged.connect(self.uncheck)
        self.checkBox5.move(15,100)
        self.checkBox5.resize(320,40)

        self.checkBox6 = QCheckBox("6", self)
        self.checkBox6.stateChanged.connect(self.uncheck)
        self.checkBox6.move(15,120)
        self.checkBox6.resize(320,40)

        self.checkBox7 = QCheckBox("7", self)
        self.checkBox7.stateChanged.connect(self.uncheck)
        self.checkBox7.move(15,140)
        self.checkBox7.resize(320,40)

        self.checkBox8 = QCheckBox("8", self)
        self.checkBox8.stateChanged.connect(self.uncheck)
        self.checkBox8.move(15,160)
        self.checkBox8.resize(320,40)

        self.checkBox9 = QCheckBox("9", self)
        self.checkBox9.stateChanged.connect(self.uncheck)
        self.checkBox9.move(15,180)
        self.checkBox9.resize(320,40)

        self.checkBox0 = QCheckBox("0", self)
        self.checkBox0.stateChanged.connect(self.uncheck)
        self.checkBox0.move(15,200)
        self.checkBox0.resize(320,40)

        self.checkBoxAll = QCheckBox("All", self)
        self.checkBoxAll.stateChanged.connect(self.uncheck)
        self.checkBoxAll.move(15,220)
        self.checkBoxAll.resize(320,40)

        self.button = QPushButton('Filter', self)
        self.button.setToolTip('This shows the selected digits button')
        self.button.move(12,260)
        self.button.resize(65,25)
        self.button.clicked.connect(self.filterClicked)

        self.index = 0
        self.selectedNumber = 10
    
    def previousClicked(self):
        self.index = self.index - 1
        num_of_images = 100 # solution found from https://www.pluralsight.com/guides/building-your-first-pytorch-solution
        if self.index < 0: #if previous button is pressed when there is no previous image
            QMessageBox.about(self, "ERROR!", "There is no previous image") #show error msg
            self.index = 0
            pixmap = QtGui.QPixmap("initTest.png") #show init.png
            self.label.setPixmap(pixmap)
        else:
            if self.selectedNumber == 10:
                for index in range(1, num_of_images+1):
                    plt.subplot(10, 10, index)
                    plt.axis('off')
                    plt.imshow(TestingImages.imagesTest[(index+(100*self.index))].numpy().squeeze(), cmap='gray_r')
                plt.savefig("prev.png")
                pixmap = QtGui.QPixmap("prev.png") #show prev.png
                self.label.setPixmap(pixmap)
            else:
                for index in range(1, num_of_images+1):
                    plt.subplot(10, 10, index)
                    plt.axis('off')
                    if TestingImages.labelsTest[(index+(100*self.index))] == self.selectedNumber:
                        plt.imshow(TestingImages.imagesTest[(index+(100*self.index))].numpy().squeeze(), cmap='gray_r')
                    else:
                        plt.axis('off')
                        white = np.ones((28, 28), dtype=np.float)
                        plt.imshow(white, cmap='gray',vmin=0,vmax=1) #show blank image - https://stackoverflow.com/questions/28234416/plotting-a-white-grayscale-image-in-python-matplotlib
                plt.savefig("prev.png")
                pixmap = QtGui.QPixmap("prev.png") #show prev.png
                self.label.setPixmap(pixmap)

    def nextClicked(self):
        self.index = self.index + 1
        num_of_images = 100 # solution found from https://www.pluralsight.com/guides/building-your-first-pytorch-solution
        if self.index == 599:
            QMessageBox.about(self, "ERROR!", "There is no previous image") #show error msg
            self.index = 0
            pixmap = QtGui.QPixmap("next.png") #show the existing image
            self.label.setPixmap(pixmap)
        else:
            if self.selectedNumber ==10: #if the person has selected "show all" option
                for index in range(1, num_of_images+1):
                    plt.subplot(10, 10, index)
                    plt.axis('off')
                    plt.imshow(TestingImages.imagesTest[(index+(100*self.index))].numpy().squeeze(), cmap='gray_r')
                plt.savefig("prev.png")
                pixmap = QtGui.QPixmap("prev.png") #show prev.png
                self.label.setPixmap(pixmap)
            else:
                for index in range(1, num_of_images+1):
                    plt.subplot(10, 10, index)
                    plt.axis('off')
                    if TestingImages.labelsTest[(index+(100*self.index))] == self.selectedNumber:
                        plt.imshow(TestingImages.imagesTest[(index+(100*self.index))].numpy().squeeze(), cmap='gray_r')
                    else:
                        plt.axis('off')
                        white = np.ones((28, 28), dtype=np.float)
                        plt.imshow(white, cmap='gray',vmin=0,vmax=1)
                plt.savefig("next.png")
                pixmap = QtGui.QPixmap("next.png") #show next.png 
                self.label.setPixmap(pixmap)

    def filterClicked(self):
        num_of_images = 100
        if self.selectedNumber == 10: #if the person has selected "show all" option
            for index in range(1, num_of_images+1):
                plt.subplot(10, 10, index)
                plt.axis('off')
                plt.imshow(TestingImages.imagesTest[(index+(100*self.index))].numpy().squeeze(), cmap='gray_r')
            plt.savefig("init.png")
            pixmap = QtGui.QPixmap("init.png") #show prev.png
            self.label.setPixmap(pixmap)
        else: #if the person presses one of the number options
            for index in range(1, num_of_images+1):
                plt.subplot(10, 10, index)
                plt.axis('off')
                if TestingImages.labelsTest[(index+(100*self.index))] == self.selectedNumber:
                    plt.imshow(TestingImages.imagesTest[(index+(100*self.index))].numpy().squeeze(), cmap='gray_r')
                else:
                    plt.axis('off')
                    white = np.ones((28, 28), dtype=np.float)
                    plt.imshow(white, cmap='gray',vmin=0,vmax=1)
            plt.savefig("init.png")
            pixmap = QtGui.QPixmap("init.png") #show next.png 
            self.label.setPixmap(pixmap)

    def uncheck(self, state): #https://www.geeksforgeeks.org/pyqt5-selecting-any-one-check-box-among-group-of-check-boxes/
        if state == Qt.Checked:
            # if first check box is selected
            if self.sender() == self.checkBox1:
                self.selectedNumber = 1
                # making other check box to uncheck
                self.checkBox2.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)


            # if second check box is selected
            elif self.sender() == self.checkBox2:
                self.selectedNumber = 2
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)
            
            # if third check box is selected
            elif self.sender() == self.checkBox3:
                self.selectedNumber = 3
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)
            
            # if fourth check box is selected
            elif self.sender() == self.checkBox4:
                self.selectedNumber = 4
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)
            
            # if fifth check box is selected
            elif self.sender() == self.checkBox5:
                self.selectedNumber = 5
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)
            
            # if sixth check box is selected
            elif self.sender() == self.checkBox6:
                self.selectedNumber = 6
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)
            
            # if seventh check box is selected
            elif self.sender() == self.checkBox7:
                self.selectedNumber = 7
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)
            
            # if eighth check box is selected
            elif self.sender() == self.checkBox8:
                self.selectedNumber = 8
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)

            # if nineth check box is selected
            elif self.sender() == self.checkBox9:
                self.selectedNumber = 9
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox0.setChecked(False)
                self.checkBoxAll.setChecked(False)

            # if zeroth check box is selected
            elif self.sender() == self.checkBox0:
                self.selectedNumber = 0
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBoxAll.setChecked(False)

            elif self.sender() == self.checkBoxAll:
                self.selectedNumber = 10
                # making other check box to uncheck
                self.checkBox1.setChecked(False)
                self.checkBox3.setChecked(False)
                self.checkBox4.setChecked(False)
                self.checkBox5.setChecked(False)
                self.checkBox6.setChecked(False)
                self.checkBox7.setChecked(False)
                self.checkBox8.setChecked(False)
                self.checkBox9.setChecked(False)
                self.checkBox2.setChecked(False)
                self.checkBox0.setChecked(False)

    def closeEvent(self, event): # remove the created images
        if os.path.exists("prev.png"):
            os.remove("prev.png")

        if os.path.exists("next.png"):
            os.remove("next.png")
