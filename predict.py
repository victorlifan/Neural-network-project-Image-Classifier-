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
import json
##############################################################
##############################################################

# Initiate variables with default values
#checkpoint = 'checkpoint.pth'
#filepath = 'cat_to_name.json'
#arch=''
#topk = 5

# generate random image_path
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
path=test_dir+'/'+np.random.choice(os.listdir(test_dir))
image=np.random.choice(os.listdir(path))
image_path=path+'/'+image


# Set up parameters for entry in command line
parser = argparse.ArgumentParser(description='Image Classifier Neural network predicting section')
parser.add_argument("image_path",help="Path to image (default:random generate through code)",type=str, default=image_path,nargs='?')
parser.add_argument('-c','--checkpoint', action='store',type=str, help='Name of trained model to be loaded and used for predictions.', default="checkpoint.pth")
parser.add_argument('-k', '--topk', action='store',type=int, help='Select number of classes you wish to see in descending order. (default:top 5)', default=5)
parser.add_argument('-j', '--json', action='store',type=str, help='Define name of json file holding class names.', default= "cat_to_name.json")
parser.add_argument('-g','--gpu', action='store_true', help='Use GPU if available', default='cpu')

args = parser.parse_args()

# Select parameters entered in command line
if args.checkpoint:
    checkpoint = args.checkpoint
if args.image_path:
    image_path = args.image_path
if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

##############################################################
##############################################################

def load_model(checkpoint_path):
    '''
    load model from a checkpoint
    '''
    checkpoint=torch.load(checkpoint_path)
    model=checkpoint['model_type']
    leaning_rate=checkpoint['leaning_rate']
    model.epochs=checkpoint['epoch']
    model.classifier=checkpoint['classifer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx =checkpoint['class_to_idx']
    optimizer=checkpoint['optimizer']
    for parameter in model.parameters():
        parameter.requires_grad=True
    model.eval()
    return model

model= load_model(checkpoint)
#print(model)
print("="*20+"MODEL LOADED"+"="*20)

##############################################################
##############################################################

### preprocess images for prediction

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im= Image.open(image_path)
    im= im.resize((256,256))
    cropvalue=(256-224)/2
    im=im.crop((cropvalue,cropvalue,256-cropvalue,256-cropvalue))
    im= np.array(im)/255
    im=(im-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    im= im.transpose((2,0,1))
    return im

##############################################################
##############################################################

def predict(image_path, model, topk=topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    # prefix image
    im=process_image(image_path)

    '''uncomment to turn on or off the image show'''
    #imshow(im)
    testim=torch.from_numpy(im)
    # change shape to 4 dimensions [1,3,224,224]
    testim=testim.unsqueeze_(0)
    testim=testim.float()
    # invert indice to class
    inv_index={v:k for k,v in model.class_to_idx.items()}

    # get top 5 ps
    output=model.forward(testim)
    ps=torch.exp(output)
    #convert into list
    pros,indexlist =ps.topk(topk,dim=1)[0].tolist()[0],ps.topk(topk,dim=1)[1].tolist()[0]
    # map class back to probabilities
    classes=[inv_index[i] for i in indexlist]

    # map names
    names = [cat_to_name[i] for i in classes]
    truth=names[0]

    return truth, names, pros

truth, names, pros= predict(image_path,model,topk)

##############################################################
##############################################################

i= 0
while i <topk:
    print("{} with a probability of {}".format(names[i], pros[i]))
    i +=1
