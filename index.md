## Welcome to Image Captioning with Keras Project

### Introduction

### Dataset Description

This project is based on Flick8k dataset from Kaggle which is a subset of Flickr30k. Flick8k contains a total of 8092 images in JPEG format with different shapes and sizes, 6000 of which are used for training, 1000 for testing and another 1000 for development. In our project, we treated 6000 instances for training and 1000 instances for testing, which we split ourselves from the data. We did not use the given sets for training, test and development.

The dataset is available at https://www.kaggle.com/shadabhussain/flickr8k

### Data Preparation

Since this is an image captioning project, there are two elements - Images and Texts. Therefore, we need to prepare both in order to train a model that could learn to generate captions by identifying what lies in the image.

#### Image Feature Extraction

The first part of this project is extracting features of images using pre-trained CNN models for image recognition. We used VGG16, VGG19 and Inception_Resnet_V2 for comparing performances for these three CNN architectures trained on ImageNet.

1. VGG16

Created in 2014, VGG16 is a Convolutional Neural Network (CNN) model used for image recognition. The main differences with previously developed CNN models lie in its architecture: it uses a combination of very small convolution filters (3x3 pixels) with a very deep network containing 16 layers for weights and parameter learning, in contrast with previous CNN models like AlexNet, which used larger convolution filter kernel sizes but fewer layers.

Convolution filters are xxx

The VGG16 architecture is detailed in the diagram below:

There are 13 convolutional layers (in black) and 3 Dense (i.e. fully connected) layers.
The max pooling layers (in red) are there to obtain informative features from the convolutional filters’ output at a series of stages in the architecture. The final softmax layer determines which class the image belongs to.

For this project, we implemented VGG16 using Keras and the default weights available from its pre-training on the ImageNet image dataset. The first step was to reshape the images to fit the (224px x 224px x 3 channels) input size required by the model before applying the preprocessing function from Keras that works specifically with VGG16. Once the pre-processing steps were completed, the images were ready to use with the VGG16 model.

While this model is normally used for image classification tasks (with 1000 classes available as output), we held off on the final layer and instead used the features extracted at the penultimate layer as input to the final image captioning model.

One of the drawbacks of this model is its large size, which I was unable to run on my own machine. Instead, I executed it using a Google Colab notebook and it took approx. 2 hours to complete (with the Flickr8k image dataset).


2. VGG19

3. Inception Resnet 





**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/women-in-ai-ireland/August-2020-WaiLEARN-Image-Caption-Generation/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
