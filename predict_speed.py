import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data_utils
from torch.autograd import Variable
from PIL import Image
import os

def prepare_dataset():
    with open('data/train.txt', 'r') as target:
        y_train  = []
        x_train = None
        x_test = None
        for line in target:
            y_train.append(float(line.strip()))
        y_train = np.asarray(y_train)

        no_of_images = len(os.listdir('data/train'))
        train_images = []
        for x in range(no_of_images):
            rgb = np.array(Image.open('data/train/train_'+str(x)+'.png'))
            train_images.append(rgb)
        print(train_images)
    return x_train, y_train, x_test

x_train, y_train, x_test = prepare_dataset()
print(y_train)
