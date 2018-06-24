import copy
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class sun_dataset (torch.utils.data.Dataset):
    
    def __init__ (self, txt_file, transform=None):
        super().__init__()
        self.df = pd.read_csv(txt_file, header=None, sep=' ')
        self.transform = transform
        print('Loading from %s' % txt_file)
        
    def __len__ (self):
        return len(self.df)
    
    def __getitem__ (self, idx):
        
        image = Image.open(self.df.iloc[idx, 0])
        label = self.df.iloc[idx, 1] - 1
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    
class MyResNet (nn.Module):
    
    def __init__ (self, pretrained_resnet, num_classes, drop_rate, fc_add_dim):
        super().__init__()
        self.pretrained_resnet = pretrained_resnet
        
        # Reset the fc layer
        self.num_features = self.pretrained_resnet.fc.in_features
        
        self.fc_add = nn.Linear(self.num_features, fc_add_dim)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(fc_add_dim, num_classes)
        
        print('Intermediate fc dimension is: %d' % fc_add_dim)
        print('Dropout rate is: %f' % drop_rate)
        
    def forward(self, x):
        x = self.pretrained_resnet.conv1(x)
        x = self.pretrained_resnet.bn1(x)
        x = self.pretrained_resnet.relu(x)
        x = self.pretrained_resnet.maxpool(x)

        x = self.pretrained_resnet.layer1(x)
        x = self.pretrained_resnet.layer2(x)
        x = self.pretrained_resnet.layer3(x)
        x = self.pretrained_resnet.layer4(x)

        x = self.pretrained_resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc_add(x))
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
       
def train_model (model, dataloaders, dataset_sizes, batch_size, num_classes, loss_function, 
                 optimizer, scheduler, num_epochs, device, display_step, log_dir, model_id=None):
    
    # Deep copy model weights
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Start training
    training_step = 0
    for epoch in range(num_epochs):
          
        # Loop over training phase and validation phase
        for phase in ['train', 'val']:
            
            # Set model modes and set scheduler
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_correct_total = 0
            
            class_correct = torch.tensor([0. for i in range(num_classes)])
            class_total = torch.tensor([0. for i in range(num_classes)])
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # If on training phase, enable gradients
                with torch.set_grad_enabled(phase == 'train'):
                    
                    logits = model(inputs)
                    _, preds = torch.max(logits, 1)
                    loss = loss_function(logits, labels)
                    
                    # Backward if training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        training_step += 1
                        
                        if training_step % display_step == 0:
                            minibatch_loss = loss.item()
                            minibatch_acc = (preds == labels).sum().item() / batch_size
                            print('Epoch: [%d/%d], Step: %5d, Minibatch_loss: %.3f, Minibatch_accuracy_micro: %.3f' 
                                  % (epoch, num_epochs, training_step, minibatch_loss, minibatch_acc))
                        
                # Record loss and correct predictions
                correct_tensor = (preds == labels).squeeze()
                running_loss += loss.item() * inputs.shape[0]
                running_correct_total += correct_tensor.sum().item()
                
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += correct_tensor[i].item()
                    class_total[label] += 1
                
            # Epoch loss and accuracieds
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc_mic = running_correct_total / dataset_sizes[phase]
            epoch_acc_mac = (class_correct / class_total).mean().item()
            
            print('Epoch: [%d/%d], Phase: %s, Epoch_loss: %.3f, Epoch_accuracy_micro: %.3f, Epoch_accuracy_macro: %.3f' 
                  % (epoch, num_epochs, phase, epoch_loss, epoch_acc_mic, epoch_acc_mac))
            
            # Deep copy the best model weights
            if phase == 'val' and epoch_acc_mic > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc_mic
                best_model_weights = copy.deepcopy(model.state_dict())
                
    print()
    print('Training Complete.')
    print('Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch))
    
    # Load the best model weights
    model.load_state_dict(best_model_weights)
    
    # Save the best model
    model_states = {'epoch': epoch + 1,
                    'best_epoch': best_epoch,
                    'state_dict_best': best_model_weights,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()}
    
    torch.save(model_states, os.path.join(log_dir, 'model_%s_checkpoint.pth.tar' % model_id))
    
    return model