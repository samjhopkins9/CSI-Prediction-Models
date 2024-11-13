#!/usr/bin/env bash
# Set working directory for file output (if not set, will be directory from which script is run)
wd="."
# Change to output directory and create out folder if it doesn't already exist
cd "$wd"
if [ ! -d out ]; then
  mkdir out
fi

pyProgram=$(cat <<EOF
#!/usr/bin/python
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import random
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import seaborn as sn
import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

tf.keras.backend.clear_session()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

train_dl_origin = torch.load('Dataset/train_dl.pt')
valid_dl_origin = torch.load('Dataset/valid_dl.pt')

train_CSI = train_dl_origin.dataset[:][0]
train_label = train_dl_origin.dataset[:][1][:,0:2]
train_label1 = train_dl_origin.dataset[:][1][:,2].type(torch.LongTensor)

valid_CSI = valid_dl_origin.dataset[:][0]
valid_label = valid_dl_origin.dataset[:][1][:,0:2]
valid_label1 = valid_dl_origin.dataset[:][1][:,2].type(torch.LongTensor)

train_CSI_modulus = torch.abs(train_CSI)
valid_CSI_modulus = torch.abs(valid_CSI)


# correcting shape of class label tensors, ensuring proper data types for all tensors
train_label1 = train_label1.reshape(-1, 1)
valid_label1 = valid_label1.reshape(-1, 1)
train_label = tf.convert_to_tensor(train_label, dtype=tf.float32)
valid_label = tf.convert_to_tensor(valid_label, dtype=tf.float32)
train_label1 = tf.convert_to_tensor(train_label1, dtype=tf.float32)
valid_label1 = tf.convert_to_tensor(valid_label1, dtype=tf.float32)

# correcting shape of input tensors
train_CSI_modulus = tf.squeeze(train_CSI_modulus)
valid_CSI_modulus = tf.squeeze(valid_CSI_modulus)


# Normalization
train_min = tf.reduce_min(train_CSI_modulus)
train_max = tf.reduce_max(train_CSI_modulus)

train_CSI_modulus = (train_CSI_modulus - train_min) / (train_max - train_min)

valid_min = tf.reduce_min(valid_CSI_modulus)
valid_max = tf.reduce_max(valid_CSI_modulus)

valid_CSI_modulus = (valid_CSI_modulus - valid_min) / (valid_max - valid_min)



def RSquared(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))  # Residual sum of squares
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))  # Total sum of squares
    return 1 - ss_res / (ss_tot + K.epsilon())



# CNN Model
inputLayer = Input(shape=(4, 1632))

convLayer1 = Conv1D(filters=16, kernel_size=1, activation='relu', name='conv1')(inputLayer)
convLayer2 = Conv1D(filters=16, kernel_size=4, activation='relu', name='conv2')(convLayer1)
flattenedLayer = Flatten()(convLayer2)
denseLayer = Dense(128, activation='sigmoid')(flattenedLayer)


XY_pos = Dense(2, name="XY_Position")(denseLayer)
LoS_nLoS = Dense(1, activation="sigmoid", name="LoS_nLoS")(denseLayer)

posModel = Model(inputs=inputLayer, outputs=XY_pos)
losModel = Model(inputs=inputLayer, outputs=LoS_nLoS)

posModel.compile(
    optimizer='adam',
    loss = 'mse',
    metrics=[RSquared]
)

posHistory = posModel.fit(

    train_CSI_modulus,
    train_label,
    validation_data=(
        valid_CSI_modulus,
        valid_label
    ),
    epochs = 100,
    batch_size = 128
)

losModel.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']

)

losHistory = losModel.fit(

    train_CSI_modulus,
    train_label1,
    validation_data=(
    
        valid_CSI_modulus,
        valid_label1
    ),
    epochs=100,
    batch_size=128

)



EOF
)

pyProgram_dataWrite=$(cat <<EOF

#!/usr/bin/python
import torch
import torch.nn as nn
import numpy as np
import scipy.io
import random
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import seaborn as sn
import pandas as pd

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, concatenate
import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

tf.keras.backend.clear_session()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

train_dl_origin = torch.load('Dataset/train_dl.pt')
valid_dl_origin = torch.load('Dataset/valid_dl.pt')

train_CSI = train_dl_origin.dataset[:][0]
train_label = train_dl_origin.dataset[:][1][:,0:2]
train_label1 = train_dl_origin.dataset[:][1][:,2].type(torch.LongTensor)

valid_CSI = valid_dl_origin.dataset[:][0]
valid_label = valid_dl_origin.dataset[:][1][:,0:2]
valid_label1 = valid_dl_origin.dataset[:][1][:,2].type(torch.LongTensor)

train_CSI_modulus = torch.abs(train_CSI)
valid_CSI_modulus = torch.abs(valid_CSI)


# data preprocessing for C++
def consolidate_antenna(x):
    l0 = []

    for i in range(len(x)):
        l0.append([])
        for j in range(len(x[i])):
            for k in range(len(x[i][j])):
                mean = torch.mean(x[i][j][k], dim=None, keepdim=False)
                l0[i].append(mean)

    return torch.tensor(l0)
    

x_train = tf.convert_to_tensor(consolidate_antenna(train_CSI_modulus), dtype=tf.float32)
x_test = tf.convert_to_tensor(consolidate_antenna(valid_CSI_modulus), dtype=tf.float32)
x_train_large = train_CSI_modulus.view(15000, -1)
x_test_large = valid_CSI_modulus.view(5000, -1)

x_train_np = x_train.numpy()
x_test_np = x_test.numpy()
x_train_large_np = x_train_large.numpy()
x_test_large_np = x_test_large.numpy()
y_train0_np = train_label.numpy()
y_test0_np = valid_label.numpy()
y_train1_np = train_label1.numpy()
y_test1_np = valid_label1.numpy()

np.savetxt('x_train.txt', x_train_np)
np.savetxt('x_test.txt', x_test_np)
np.savetxt('x_train_large.txt', x_train_large_np)
np.savetxt('x_test_large.txt', x_test_large_np)
np.savetxt('y_train_0.txt', y_train0_np)
np.savetxt('y_test_0.txt', y_test0_np)
np.savetxt('y_train_1.txt', y_train1_np)
np.savetxt('y_test_1.txt', y_test1_np)

EOF
)


# C++
cppProgram=$(cat <<EOF

#include "../ProcessingFunctions.h"
#include "../MachineLearning.h"
#include "../MultilayerPerceptron.h"
#include "../ConvolutionalNeuralNet.h"
#include <chrono>

int main(){

    auto startTime = chrono::high_resolution_clock::now();

    DataFrame x_train = DataFrame();
    DataFrame x_test = DataFrame();
    DataFrame x_train_large = DataFrame();
    DataFrame x_test_large = DataFrame();
    DataFrame y_train_0 = DataFrame();
    DataFrame y_test_0 = DataFrame();
    DataFrame y_train_1 = DataFrame();
    DataFrame y_test_1 = DataFrame();
    
    x_train.readTxt("x_train.txt");
    x_test.readTxt("x_test.txt");
    x_train_large.readTxt("x_train_large.txt");
    x_test_large.readTxt("x_test_large.txt");
    x_train_large.readTxt_3D("x_train_large.txt", 4, 1632);
    x_test_large.readTxt_3D("x_test_large.txt", 4, 1632);
    x_train.MinMaxNormalize();
    x_test.MinMaxNormalize();
    x_train_large.MinMaxNormalize();
    x_test_large.MinMaxNormalize();
    x_train_large.MinMaxNormalize_3D();
    x_test_large.MinMaxNormalize_3D();
    // x_test_large.print_3D();
    
    // y for regression
    y_train_0.readTxt("y_train_0.txt");
    y_test_0.readTxt("y_test_0.txt");
    vector<double> y_xpos_train = y_train_0.getColumn(0);
    vector<double> y_xpos_test = y_test_0.getColumn(0);
    vector<double> y_ypos_train = y_train_0.getColumn(1);
    vector<double> y_ypos_test = y_test_0.getColumn(1);
    
    // y for classification
    y_train_1.readTxt("y_train_1.txt");
    y_test_1.readTxt("y_test_1.txt");
    vector<double> y_class_train = y_train_1.getColumn(0);
    vector<double> y_class_test = y_test_1.getColumn(0);
    
    
    
    
    // regression (ML)
    LinearRegression linearModel = LinearRegression();
    linearModel.train(x_train, y_xpos_train, 1000, 0.5);
    
    vector<double> y_xpos_pred_train = linearModel.predictions;
    double train_error = MSE(y_xpos_pred_train, y_xpos_train);
    double train_R2 = RSquared(y_xpos_pred_train, y_xpos_train);
    cout << "Linear Regression Train MSE for X: " << train_error << endl;
    cout << "Linear Regression Train R Squared for X: " << train_R2 << endl << endl;
    
    vector<double> y_xpos_pred_test = linearModel.predict(x_test);
    double test_error = MSE(y_xpos_pred_test, y_xpos_test);
    double test_R2 = RSquared(y_xpos_pred_test, y_xpos_test);
    cout << "Linear Regression Test MSE for X: " << test_error << endl;
    cout << "Linear Regression Test R Squared for X: " << test_R2 << endl << endl;
    
    linearModel.train(x_train, y_ypos_train, 1000, 0.5);
    
    vector<double> y_ypos_pred_train = linearModel.predictions;
    double train_error1 = MSE(y_ypos_pred_train, y_ypos_train);
    double train_R21 = RSquared(y_ypos_pred_train, y_ypos_train);
    cout << "Linear Regression Train MSE for Y: " << train_error1 << endl;
    cout << "Linear Regression Train R Squared for Y: " << train_R21 << endl << endl;
    
    vector<double> y_ypos_pred_test = linearModel.predict(x_test);
    double test_error1 = MSE(y_ypos_pred_test, y_ypos_test);
    double test_R21 = RSquared(y_ypos_pred_test, y_ypos_test);
    cout << "Linear Regression Test MSE for Y: " << test_error1 << endl;
    cout << "Linear Regression Test R Squared for Y: " << test_R21 << endl << endl;
    
    
    
    
    // classification (ML)
    KNN knnModel = KNN(x_train, y_class_train, 7);
    vector<double> y_pred_train = knnModel.predict(x_train);
    double train_acc = Accuracy(y_pred_train, y_class_train);
    vector<double> y_pred_test = knnModel.predict(x_test);
    double test_acc = Accuracy(y_pred_test, y_class_test);
    cout << "KNN Training Accuracy: " << train_acc << endl;
    cout << "KNN Test Accuracy: " << test_acc << endl;
    
    LogisticRegression logisticModel = LogisticRegression();
    logisticModel.train(x_train, y_class_train, 1000, 0.01);
    
    vector<double> y_pred_train1 = logisticModel.predictions;
    double train_acc1 = Accuracy(y_pred_train1, y_class_train);
    vector<double> y_pred_test1 = logisticModel.predict(x_test);
    double test_acc1 = Accuracy(y_pred_test1, y_class_test);
    cout << "Logistic Regression Training Accuracy: " << train_acc1 << endl;
    cout << "Logistic Regression Test Accuracy: " << test_acc1 << endl;
    
    
    
    /*
    // regression (MLP)
    int HL_dim = 3;
    int HL_size = 5;
    double learningRate = 0.000001;
    double clipThreshold = 6.0;
    
    MLP perceptron = MLP(x_train_large, y_xpos_train, HL_size, HL_dim, learningRate, clipThreshold);
    
    int iterations = 5;
    perceptron.train(iterations, true);
    double train_mse_perceptron = MSE(perceptron.predicted, y_xpos_train);
    double train_R2_perceptron = RSquared(perceptron.predicted, y_xpos_train);
    cout << train_mse_perceptron << endl;
    cout << train_R2_perceptron << endl;
    // perceptron.printOutput();
    */
    
    
    /*
    // regression (CNN)
    int CL_dim = 2;
    int filterSize = 150;
    int FCL_dim = 1;
    int FCL_size = 5;
    int stride = 1;
    int poolStride = 3;
    double learningRate = 0.000001;
    double clipThreshold = 6.0;
    
    CNN conv = CNN(x_train_large, y_xpos_train, CL_dim, filterSize, FCL_dim, FCL_size, stride, poolStride, learningRate, clipThreshold);
    
    int iterations = 3;
    conv.train(iterations);
    double train_MSE_conv = MSE(conv.predicted, y_xpos_train);
    double train_R2_conv = RSquared(conv.predicted, y_xpos_train);
    cout << train_MSE_conv << endl;
    cout << train_R2_conv << endl;
    */
    
    
    
    // running time
    auto endTime = chrono::high_resolution_clock::now();
    auto runTime = chrono::duration_cast<chrono::nanoseconds>(endTime - startTime);
    cout << "run time: " << runTime.count() * pow(10, -9) << " seconds" << endl;
    
    
    
    return 0;

}


EOF
)


# Run scripts

function runPy {

    echo "$pyProgram" > out/Project1.py
    python3 out/Project1.py

}

function run_cPP {
  echo "$cppProgram" > out/Project1.cpp
  echo "$pyProgram_dataWrite" > out/Project1_writeTxt.py
  python3 out/Project1_writeTxt.py # comment/uncomment as needed
  g++ --std=c++17 out/Project1.cpp
  ./a.out
  rm a.out
}

# Call functions: comment/uncomment based on which scripts you want to run
runPy
# run_cPP
