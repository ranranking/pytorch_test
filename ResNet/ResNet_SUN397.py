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
parser.add_argument('--dataset', default='plain', type=str)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--model_id', default='plain', type=str)
parser.add_argument('--decay_epoch', default=30, type=int)
parser.add_argument('--dropout', default=False)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--fc_add_dim', default=4096, type=int)
parser.add_argument('--caffe_weights', default=False)
args = parser.parse_args()


# ============================================
# Misc
LOG_DIR = './log'
DATASET = args.dataset
MODEL_ID = args.model_id
DATALOADER_WORKERS = 4
LEARNING_RATE = 0.01
LR_DECAY_FACTOR = 0.1
LR_DECAY_EPOCHS = args.decay_epoch
DROPOUT = args.dropout
DROPOUT_RATE = args.dropout_rate
FC_ADD_DIM = args.fc_add_dim
MOMENTUM = 0.9
EPOCHS = args.epoch
BATCH_SIZE = 256 
DISPLAY_STEP = 10
NUM_CLASSES = 397
CAFFE_WEIGHTS = args.caffe_weights

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
    ])
}

print('Load datasets.')

datasets = {x: sun_dataset(txt_file='./%s_txt/sun397_%s_lt.txt' % (DATASET, x), transform=data_transforms[x]) 
            for x in ['train', 'val']} 
dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS) 
               for x in ['train', 'val']}
dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

print('Done.')



# ============================================
# Model Preparation

print('Load pretrained ResNet.')

# Load pretrained model
if CAFFE_WEIGHTS:
    print('Model using caffe pretrained weights.')
    resnet = torchvision.models.resnet152(pretrained=False)
    weights = torch.load('./caffe_weights/resnet152.pth')
    resnet.load_state_dict(weights)
else:
    print('Model using pytorch pretrained weights.')
    resnet = torchvision.models.resnet152(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

print('Done.')    


if DROPOUT:
    print('Model using dropout.')
    resnet = MyResNet(resnet, NUM_CLASSES, DROPOUT_RATE, FC_ADD_DIM)
    # Optimizer only on the last fc layers
    optimizer = optim.SGD(list(resnet.fc.parameters()) + list(resnet.fc_add.parameters()), lr=LEARNING_RATE, momentum=MOMENTUM)
else:
    print('Model not using dropout.')
    # Reset the fc layer
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, NUM_CLASSES)
    # Optimizer only on the last fc layers
    optimizer = optim.SGD(list(resnet.fc.parameters()), lr=LEARNING_RATE, momentum=MOMENTUM)

# Loss function 
loss_function = nn.CrossEntropyLoss()

# Decay LR 
print('Learning rate decay epoch: %s' % LR_DECAY_EPOCHS)
print('Learning rate decay factor: %s' % LR_DECAY_FACTOR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EPOCHS, gamma=LR_DECAY_FACTOR)

resnet = resnet.to(device)



# ============================================
# Training
print('Start training')
resnet = train_model(model=resnet, dataloaders=dataloaders, batch_size=BATCH_SIZE,
                     dataset_sizes=dataset_sizes, num_classes=NUM_CLASSES, loss_function=loss_function, 
                     optimizer=optimizer, scheduler=exp_lr_scheduler, 
                     device=device, num_epochs=EPOCHS, display_step=DISPLAY_STEP,
                     log_dir=LOG_DIR, model_id=MODEL_ID)
