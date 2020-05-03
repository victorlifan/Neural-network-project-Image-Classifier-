# -*- coding: utf-8 -*-
# Imports here
from time import time
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import os
import seaborn as sns
import argparse


##############################################################
##############################################################

# Set up parameters for entry in command line
parser =argparse.ArgumentParser(description='Image Classifier Neural network training section')
parser.add_argument('data_dir',type=str,help='Location of directory with data for image classifier to train and test',default="./flowers/",nargs='*')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose among 3 pretrained networks - densenet161, alexnet, and vgg16 (default:densenet161)', default='densenet161')
parser.add_argument('-H','--hidden_units',action='store',type=int, help='Select number of hidden units for 1st layer',default=1104)
parser.add_argument('-l','--learning_rate',action='store',type=float, help='Choose a float number as the learning rate for the model (default:0.02, suggestion: 0.01,0.02,0.03)',default= 0.002)
parser.add_argument('-e','--epochs',action='store',type=int, help='Choose the number of epochs you want to perform gradient descent (default:10)',default=10)
parser.add_argument('-s','--save_dir',action='store', type=str, help='Select name of file to save the trained model')
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available',default="gpu")

args = parser.parse_args()
# Select parameters entered in command line
if args.arch:
    arch = args.arch
if args.hidden_units:
    hidden_units = args.hidden_units
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

##############################################################
##############################################################

def create_model(arch='densenet161',hidden_units=hidden_units,learning_rate=learning_rate):
    '''
    Function builds model
    '''
    # Select from available pretrained models
    model =  getattr(models,arch)(pretrained=True)

    # model selection
    if arch == 'densenet161':
        in_features= model.classifier.in_features
    elif arch == 'vgg16':
        in_features= model.classifier[0].in_features
    elif arch == 'alexnet':
        in_features= model.classifier[1].in_features


    #Freeze feature parameters so as not to backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # define classifier
    classifier= nn.Sequential(nn.Linear(in_features, hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(p=.3),
                                   nn.Linear(hidden_units,552),
                                   nn.ReLU(),
                                   nn.Dropout(p=.3),
                                   nn.Linear(552,102),
                                   nn.LogSoftmax(dim=1))

    model.classifier =classifier
    # criterion and optimizer
    criterion= nn.NLLLoss()
    optimizer= optim.Adam(model.classifier.parameters(),lr=learning_rate)


    return model,criterion,optimizer,in_features

model, criterion,optimizer,in_features=create_model(arch, hidden_units, learning_rate)

print("="*20+"MODEL IS BUILT"+"="*20)

##############################################################
##############################################################
# Directory location of images
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# data transformations
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms= transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

# TODO: Load the datasets with ImageFolder
train_data=datasets.ImageFolder(train_dir, transform= train_transforms)
validation_data= datasets.ImageFolder(valid_dir, transform=test_transforms)
test_data = datasets.ImageFolder(test_dir, transform= test_transforms)

image_datasets=datasets
# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
testloader=torch.utils.data.DataLoader(test_data, batch_size=64)


##############################################################
##############################################################

# model train function
def train_model(model, criterion, optimizer, epochs=epochs):
    '''
    Function that trains pretrained model and classifier on image dataset and validates.
    '''
    print("="*20+"STRAT TRAINING (This will take a while)"+"="*20)
    # training and validate
    model.to(device)
    epochs=epochs
    time0=time()
    for e in range(epochs):
        step=0
        running_loss=0
        for traininput, trainlabel in trainloader:
            #setup
            step+=1
            traininput, trainlabel=traininput.to(device), trainlabel.to(device)
            optimizer.zero_grad()

            # implementation
            trainoutput=model.forward(traininput)
            trainloss = criterion(trainoutput,trainlabel)
            trainloss.backward()
            optimizer.step()
            running_loss+=trainloss.item()

            # validation set run for every 50 steps
            if step % 50 == 0:
                #setups
                model.eval()
                val_loss=0
                accuracy=0
                with torch.no_grad():
                    for valinput, vallabel in validationloader:
                        valinput, vallabel= valinput.to(device), vallabel.to(device)

                        # implementation
                        valoutput = model.forward(valinput)
                        batch_loss = criterion(valoutput,vallabel)
                        val_loss += batch_loss.item()

                        # calculate accuracy
                        ps= torch.exp(valoutput)
                        equals= ps.max(1)[1]==vallabel
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                '''training mode need to be reset here, out of for loop,  otherwise it will run validation set with train mode'''
                model.train()
                print("epoch {}/{}, steps: {}/{}".format(e+1,epochs,step, len(trainloader)),
                     "training loss: {:.3f}".format(running_loss/step),
                     "validation loss: {:.3f}".format(val_loss/len(validationloader)),
                     "validation accuracy: {:.3f}".format(accuracy/len(validationloader)),
                     "duration: {:.1f}".format(time()-time0))
    return model
model_trained=train_model(model, criterion, optimizer, epochs=epochs)
print("="*20+"TRAINING FINISHED"+"="*20)
##############################################################
##############################################################

# model saving function
def save_model(model_trained):
    '''
    Function saves the trained model architecture.
    '''
    model_trained.class_to_idx=train_data.class_to_idx
    model_trained.cpu()
    save_dir = ''
    checkpoint= {'model_type':model,
            'input_size': in_features,
            'out_put_size': 102,
            'leaning_rate': learning_rate,
             'batch_size':64,
             'epoch':epochs,
             'classifer':model_trained.classifier,
             'optimizer':optimizer.state_dict(),
            'state_dict':model_trained.state_dict(),
            'class_to_idx':model_trained.class_to_idx}

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = 'checkpoint.pth'

    torch.save(checkpoint, save_dir)

save_model(model_trained)
print(model_trained)
print("="*20+"MODEL SAVED"+"="*20)

##############################################################
##############################################################
