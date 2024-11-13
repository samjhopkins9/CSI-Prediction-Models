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
