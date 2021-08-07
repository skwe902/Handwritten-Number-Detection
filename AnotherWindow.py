from __future__ import print_function
from torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torchvision
import torch.nn.functional as F
import time
from threading import Thread
import torch

import sys, os
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QMessageBox, QDialog, QWidget, QPushButton, QVBoxLayout, QLabel, QDesktopWidget
from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtGui, QtWidgets

from io import StringIO

import Net3 

class MyThread(QThread):
    #This is a helper class to ping the progress bar to update
    change_value = pyqtSignal(int)
    def run(self):
        while (AnotherWindow.pointerSelf.progress < 100): #while task not done
            #it pings every 0.1 seconds
            self.change_value.emit(int)
            time.sleep(0.1)
        self.change_value.emit(int)

class AIModel(object):

    trainEventNotCancelled = True
    since = time.time()

    # Training settings                                                                                                                                                      
    batch_size = 32 
    epoch = 30
    device = 'cuda' if cuda.is_available() else 'cpu'

    # MNIST Dataset
    train_loader = torch.utils.data.DataLoader( 
    torchvision.datasets.MNIST('mnist_data_train', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5,), (0.5,)) #https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457
                             ])),
  batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("mnist_data_test", train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5,), (0.5,))
                             ])),
  batch_size=batch_size, shuffle=True)

    model = Net3.Net3()

    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    def __init__(self):
        AIModel.pointerSelf = self

    def trainAI(): #this function runs the train() function epoch number of times
        print(f'Training MNIST Model on {AIModel.device}')
        for i in range(1, AIModel.epoch+1):
            AIModel.train(i)
        torch.save(AIModel.model, 'trainedModel.pth')

        if(AIModel.trainEventNotCancelled):    
            m, s = divmod(time.time() - AIModel.since, 60)
            print(f'Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {AIModel.device}!')
            time.sleep(0.01)

            AnotherWindow.pointerSelf.progress = (100)
            AIModel.trainEventNotCancelled = False

    def train(iteration): #https://nextjournal.com/gkoehler/pytorch-mnist this trains the model
        AIModel.model.train()
        for batch_idx, (data, target) in enumerate(AIModel.train_loader):
            data, target = data.to(AIModel.device), target.to(AIModel.device)
            AIModel.optimizer.zero_grad()
            output = AIModel.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            AIModel.optimizer.step()

            
            if(not AIModel.trainEventNotCancelled):
                break

            AnotherWindow.pointerSelf.progress = ((iteration-1) / (AIModel.epoch)) * 100
                

            if batch_idx % 10 == 0:
                print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(iteration, batch_idx * len(data), len(AIModel.train_loader.dataset), 100. * batch_idx / len(AIModel.train_loader), loss.item()))
                pg = 100. * batch_idx / len(AIModel.train_loader)

    def test(): #tests the model
        AIModel.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in AIModel.test_loader:
                data, target = data.to(AIModel.device), target.to(AIModel.device)
                output = AIModel.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() #changed size_average to reduction.
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(AIModel.test_loader.dataset)
        AnotherWindow.pointerSelf.textConsole.append(f'===========================\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(AIModel.test_loader.dataset)} '
            f'({100. * correct / len(AIModel.test_loader.dataset):.0f}%)')


class AnotherWindow(QMainWindow):
    """ produced with QT designer
    """
    def __init__(self):
        super().__init__()

        self.progress = 0
        self.trainEventNotCancelled = True

        self.setObjectName("MainWindow")
        self.setWindowIcon(QIcon(os.getcwd() + r'\App Icon.ico'))
        self.resize(664, 452)
        self.centralwidget = QtWidgets.QWidget(self)
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
        self.textConsole.setReadOnly(True)

        self.progressBar = QtWidgets.QProgressBar(self.groupBox)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_3.addWidget(self.progressBar)
        self.progressBar.setFormat('')
        self.progressBar.setAlignment(Qt.AlignCenter)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.downloadMNISTBut = QtWidgets.QPushButton(self.groupBox)
        self.startProgressBar()

        self.downloadMNISTBut.setObjectName("downloadMNISTBut")
        self.horizontalLayout.addWidget(self.downloadMNISTBut)
        self.downloadMNISTBut.clicked.connect(self.downloadMNISTButClicked)

        self.trainBut = QtWidgets.QPushButton(self.groupBox)
        self.trainBut.setObjectName("trainBut")
        self.horizontalLayout.addWidget(self.trainBut)
        self.trainBut.clicked.connect(self.trainButClicked)

        self.testBut = QtWidgets.QPushButton(self.groupBox) #added a test button that tests using mnist 10000 test images
        self.testBut.setObjectName("testBut")
        self.horizontalLayout.addWidget(self.testBut)
        self.testBut.clicked.connect(self.testButClicked)

        self.cancelBut = QtWidgets.QPushButton(self.groupBox)
        self.cancelBut.setObjectName("cancelBut")
        self.horizontalLayout.addWidget(self.cancelBut)
        self.cancelBut.clicked.connect(self.cancelButClicked)

        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        self.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        AnotherWindow.pointerSelf = self

        sys.stdout = Log(self.textConsole) # this command links the console to textConsole (type: QTextEdit)
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def startProgressBar(self):
        self.thread = MyThread()
        self.thread.change_value.connect(self.setProgressVal)
        self.thread.start()

    def setProgressVal(self):
        self.progressBar.setValue(self.progress)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("MainWindow", ""))
        self.downloadMNISTBut.setText(_translate("MainWindow", "Download MNIST"))
        self.trainBut.setText(_translate("MainWindow", "Train"))
        self.testBut.setText(_translate("MainWindow", "Test"))
        self.cancelBut.setText(_translate("MainWindow", "Cancel"))

    def cancelButClicked(self):
        AIModel.trainEventNotCancelled = False
        self.textConsole.append("Training Cancelled: Press X to close window")
        

    def downloadMNISTButClicked(self): #when download MNIST button is clicked
        batch_size = 64
        self.textConsole.append("Downloading Training Dataset...")
        AIModel.train_dataset = datasets.MNIST(root='mnist_data_train/', train=True, transform=transforms.ToTensor(), download=False)
        self.textConsole.append("Downloading Testing Dataset...")
        AIModel.test_dataset = datasets.MNIST(root='mnist_data_test/', train=False, transform=transforms.ToTensor(), download=False)
        AIModel.train_loader = data.DataLoader(dataset=AIModel.train_dataset, batch_size=batch_size, shuffle=True)
        AIModel.test_loader = data.DataLoader(dataset=AIModel.test_dataset, batch_size=batch_size, shuffle=True)
        self.textConsole.append("Done.")

        
    def trainButClicked(self):
        #initialze the task in another thread
        AIModel.trainEventNotCancelled = True
        t1 = Thread(target=AIModel.trainAI)
        t1.start()
        self.textConsole.append(f'Training Model... Epoch size: {AIModel.epoch}')

    def testButClicked(self):
        #this tests using the mnist test dataset
        AIModel.test()

    def closeEvent(self, event):
        AIModel.trainEventNotCancelled = False

    def __del__(self):
        ##this needs to be here otherwise when we open AnotherWindow again it tries to assign sys.stdout twice and recursion errors because its already assigned 
        sys.stdout = sys.__stdout__

    

class Log(QMainWindow): #credit to kosovan on stackoverflow
    def __init__(self, edit):
        AnotherWindow.out = sys.stdout
        AnotherWindow.textConsole = edit

    def write(self, message):
            AnotherWindow.out.write(message)
            if(message != "\n"): #this has to be here because Qtextedit automatically adds a new line 
                AnotherWindow.pointerSelf.progressBar.setFormat(message)
                #AnotherWindow.textConsole.verticalScrollBar().setValue(AnotherWindow.textConsole.verticalScrollBar().maximum())    

    def flush(self):
        AnotherWindow.out.flush()