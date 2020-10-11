# Write a function that loads a checkpoint and rebuilds the model
from PIL import ImageFile, Image
# import torchnet.meter.confusionmeter as cm
# import seaborn as sn
# import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob
import matplotlib.pyplot as plt
import torch
# import time
#import numpy as np
#from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# import torchvision
from collections import OrderedDict
from torch.autograd import Variable
import PIL
from torch.optim import lr_scheduler
# import copy
# import json
# import os
from os.path import exists
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True

print(device)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet152()

    # our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 2

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 1024)),
        ('relu', nn.ReLU()),
        #('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(1024, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']


# Get index to class mapping
loaded_model, class_to_idx = load_checkpoint(
    'pneumo.pth')
idx_to_class = {v: k for k, v in class_to_idx.items()}
print(idx_to_class)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485)/(0.229)
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


def predict(image_path, topk=2):
    # image_path = 'test.jpg'
    model = loaded_model
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file

    image = torch.FloatTensor(
        [process_image(Image.open(image_path).convert('RGB'))])
    model.eval()
    # print(model)
    output = model.forward(Variable(image))
    probabilities = torch.exp(output).data.numpy()[0]
    # print(pr)

    top_idx = np.argsort(probabilities)[-topk:][::-1]
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = probabilities[top_idx]
    topclass = top_class[0]
    topprobability = top_probability[0]
    print("top class is", top_class[0])
    print("top probability", top_probability[0])

    return topclass, topprobability

    # return top_probability, top_class


# predict('test.jpg')
