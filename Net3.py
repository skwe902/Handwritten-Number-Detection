import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

class Net3(nn.Module): #https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) #2d convolutional layer kernel:5x5 matrix
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5) #convolutional operation give an output of the same size as input image
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256) # fully connected layer (576 -> 256)
        self.fc2 = nn.Linear(256, 10)   # second fully connected layer that outputs our 10 labels (0~9)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # pass data through conv1 (convolutional layer) and use ReLU activation function over x
        x = F.dropout(x, p=0.5, training=self.training) # pass data through dropout p=0.5 will lead to the maximum regularization. if training=true, apply dropout.
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # applies a 2D max pooling over an input signal composed of several planes
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2)) #pooling reduces the size of the image and reduce the number of parameters in the model.
        x = F.dropout(x, p=0.5, training=self.training) 
        x = x.view(-1,3*3*64 ) # reshape the tensor (flattens it into 576)
        x = F.relu(self.fc1(x)) #first fully connected layer
        x = F.dropout(x, training=self.training)
        x = self.fc2(x) 
        return F.log_softmax(x, dim=1)   # Apply softmax to x

        #The dim parameter dictates across which dimension the softmax operations is done. 
        #Basically, the softmax operation will transform your input into a probability distribution i.e. the sum of all elements will be 1. 

        #Dropout changed the concept of learning all the weights together to learning a fraction of the weights in the network in each training iteration.