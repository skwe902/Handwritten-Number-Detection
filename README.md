# project-1-team_01
project-1-team_01 created by GitHub Classroom

### This is CompSys 302 Project 1 - "HandWritten Number Detection Project" 
Written by Andy Kweon and Vishnu Hu

This program has been trained with the MNIST dataset to detect your own handwritten digits.
The neural network used is a CNN model that uses three 2-D convolutional layers followed by two linear layers.
As activation function, ReLUs were used and for regularization, four dropouts were used to reduce overfitting.

The model was sourced from: https://www.kaggle.com/sdelecourt/cnn-with-pytorch-for-mnist

To run the program, run the "main.py" python file using a IDE/code editor such as Visual Studio Code.

You can find the report under report.pdf

The program was written/run with Python 3.8.5. Future versions of python may work.

Before running the program, make sure you have following installed in your machine:
- pytorch
- torchvision
- pyqt5
- matplotlib.pyplot
- cv2
- scipy
- pillow
- CUDA (optional - the model can be trained/run on CPU as well)

### How to Use:
The "main.py" contains the code for the MainWindow GUI (including the canvas to draw a number) and allows the user to draw an image (using a mouse) which can be checked by drawing and clicking on the "Recognise" button. If you make a mistake, press "Clear" and it should clear the image.

You can also test the model using the "Random" button which will display/check the model using a random testing image from MNIST dataset.

To train the model, click on File > Train Model > Train.

To cancel training, press "Cancel"

To test the trained model, press "Test"

To download MNIST dataset, press "Download MNIST"

If you wish to view what MNIST dataset looks like, click View > View Testing/Training Images, which will display 100 random images from testing/training dataset respectively.

To EXIT the program, press the "X" button or click File > Exit or press Ctrl + Q on keyboard.

The "AnotherWindow.py" contains the code for training/testing the model and calls the deep learning model (in "Net3.py")

The "Testing/TrainingImages.py" contains the code for showing the MNIST images


### Note:
The most recent version of this program is Ver 3.3 - Which is the final release / hand-in version of the project. 

The current trainedModel.pth has been trained with 50 epoch and 16 batch size, which was used for the demo. To change the training values, go to AnotherWindow.py and change the epoch and batch size on line 35 and 36. 

Once you train the model, the trained model data is stored in trainedModel.pth. When you press File > Train Model > Train to train the model, it overwrites the trainedModel.pth. You can rename the existing trainedModel.pth as something else if you wish to keep the existing trained data. 

You can find the other neural networks that we have tested under the Other Neural Networks folder. 
The UI files created using Qt Designer could be found under the Qt Designer UI files folder.

