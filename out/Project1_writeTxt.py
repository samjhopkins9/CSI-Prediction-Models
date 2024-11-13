
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
