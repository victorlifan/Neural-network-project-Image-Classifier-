## Project Title
Neural network project (Image Classifier)

## by Fan Li

## Date created
Project is created on May 2nd 2020.


## Description
In this project, I trained an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. I'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

<img src='assets/Flowers.png' width=500px>



## Workflow:
> PART 1
* Load and preprocess the image dataset
* Train the image classifier on your dataset
* Use the trained classifier to predict image content

> part 2
+ Train a new network on a data set with `train.py`
+ Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image `/path/to/image` and return the flower name and class probability.



## Dataset

[`flowers`](https://github.com/victorlifan/Neural-network-project-Image-Classifier-/tree/master/flowers)

 The dataset is split into three parts, training, validation, and testing.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

 ## Summary of Findings

 * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html)
 * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
 * Train the classifier layers using backpropagation using the pre-trained network to get the features
 * Track the loss and accuracy on the validation set to determine the best hyperparameters

 > Network specification

 After several test, below specifications generate the best results.

 >(classifier): Sequential(
    -  (0): Linear(in_features=2208, out_features=1104, bias=True)
    -  (1): ReLU()
    -  (2): Dropout(p=0.3, inplace=False)
    -  (3): Linear(in_features=1104, out_features=552, bias=True)
    -  (4): ReLU()
    -  (5): Dropout(p=0.3, inplace=False)
    -  (6): Linear(in_features=552, out_features=102, bias=True)
    -  (7): LogSoftmax())

 > Adam (
  -   Parameter Group 0
  -   amsgrad: False
  -   betas: (0.9, 0.999)
  -   eps: 1e-08
  -   lr: 0.002
  -   weight_decay: 0)

** Accuracy is over 82% on test set without pre-Normalize. Which will result over 95% (most of the cases) accuracy after Image Preprocessing (predicting section).


 ## About
+ [`fImage Classifier Project.ipynb`](https://github.com/victorlifan/Neural-network-project-Image-Classifier-/blob/master/Image%20Classifier%20Project.ipynb): This is the main file where I performed my work on the project. Developing an Image Classifier with Deep Learning
+ [`train.py`](https://github.com/victorlifan/Neural-network-project-Image-Classifier-/blob/master/train.py): interactive command line py file. It will train a new network on a dataset and save the model as a checkpoint.
+ [`predict.py`](https://github.com/victorlifan/Neural-network-project-Image-Classifier-/blob/master/predict.py): interactive command line py file. It uses a trained network to predict the class for an input image.
+ [`helper.py`](https://github.com/victorlifan/Neural-network-project-Image-Classifier-/blob/master/helper.py) Help to show images in [`fImage Classifier Project.ipynb`](https://github.com/victorlifan/Neural-network-project-Image-Classifier-/blob/master/Image%20Classifier%20Project.ipynb)

## Software used
+ Jupyter Notebook
+ Atom
+ Python 3.7
> + PyTorch
> + Numpy
> + Json
> + Matplotlib
> + Seaborn
> + PIL



## Credits
+ Data provided by:
    + [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
+ Instruction and assist: [Intro to Machine Learning with PyTorch](https://www.udacity.com/course/intro-to-machine-learning-nanodegree--nd229)
+ [ argparse module](https://docs.python.org/3/library/argparse.html)
+ [argparse — Command-Line Option and Argument Parsing](https://pymotw.com/3/argparse/)
