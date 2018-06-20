import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy
import argparse
from PIL import Image
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='plain', type=str)
args = parser.parse_args()


# ============================================
# Misc
LOG_DIR = './log'
MODEL_ID = args.model_id
DATALOADER_WORKERS = 4
LEARNING_RATE = 0.01
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCHS = 30
MOMENTUM = 0.9
EPOCHS = 60
BATCH_SIZE = 256
DISPLAY_STEP = 10
NUM_CLASSES = 397

if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)

print('model_id: %s' % MODEL_ID)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# ============================================
# Dataset Preparation

# transforms.RandomResizeCrop(224)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

print('Load datasets.')

datasets = {x: sun_dataset(txt_file='./%s_txt/sun397_%s_lt.txt' % (MODEL_ID, x), transform=data_transforms[x]) 
            for x in ['train', 'val', 'test']} 
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS) 
               for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val', 'test']}

print('Done.')



# ============================================
# Model Preparation

# Load pretrained model
print('Load pretrained ResNet.')
resnet = torchvision.models.resnet152(pretrained=True)
# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

print('Done.')    

# Reset the fc layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, NUM_CLASSES)

resnet = resnet.to(device)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer only on the last fc layer
optimizer = optim.SGD(resnet.fc.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Decay LR by a factor of 0.1 every 30 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)



# ============================================
# Training
print('Start training')
resnet = train_model(model=resnet, dataloaders=dataloaders, batch_size=BATCH_SIZE,
                     dataset_sizes=dataset_sizes, num_classes=NUM_CLASSES, loss_function=loss_function, 
                     optimizer=optimizer, scheduler=exp_lr_scheduler, 
                     device=device, num_epochs=EPOCHS, display_step=DISPLAY_STEP,
                     model_id=MODEL_ID)