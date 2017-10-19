# Transfer-Learning

## VGGNet

### Jianguo Zhang, April 17, 2017

Most of the time we won't want to train a whole convolutional network yourself. Modern ConvNets training on huge datasets like ImageNet take weeks on multiple GPUs. Instead, most people use a pretrained network either as a fixed feature extractor, or as an initial network to fine tune.

In this program, we'll be using [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) trained on the [ImageNet dataset](http://www.image-net.org/) as a feature extractor to classify flowers. The flower dataset comes from the [TensorFlow inception tutorial](https://www.tensorflow.org/tutorials/image_retraining).

Below is a diagram of the VGGNet architecture.


![image1](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/assets/cnnarchitecture.jpg)



![image1_1](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/vgg16.png)

VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully connected layers with our own classifier. So what we want are the values of the first fully connected layer, after being ReLUd (self.relu6). This way we can use VGGNet as a feature extractor for our images then easily train a simple classifier on top of that. What we'll do is take the first fully connected layer with 4096 units, including thresholding with ReLUs. We can use those values as a code for each image, then build a classifier on top of those codes.

 There are more informations about transfer learning from [the CS231n course notes](http://cs231n.github.io/transfer-learning/#tf).


Make sure you clone this repository into the transfer-learning directory.

`cd  transfer-learning`

`git clone https://github.com/machrisaa/tensorflow-vgg.git tensorflow_vgg`

Run `Transfer_Learning.ipynb` cell by cell or just refer the online result `Transfer_Learning.html`.

### Installation

The program using `TensorFlow==1.0.0` and `python==3.6.0 `. The `requirements.txt` describing the minimal dependencies required to run this program. 

If you have most packages, you can just install Additional Packages

`pip install tqdm`

`conda install scikit-image`

### pip

To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`.


## AlexNet

### Jianguo Zhang, April 20, 2017

AlexNet is a popular base network for transfer learning because its structure is relatively straightforward, it's not too big, and it performs well empirically.

Below is a diagram of the AlexNet architecture.

![image](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/AlexNet_image.png)


In this program, we extract AlexNet's features and use them to classify images from the [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), which includes 43 classes. The orignal AlexNet classifies for 1000 classes.

Download the [train data](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p) and [AlexNet weights](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npyhttps://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy)


Make sure the all the files are in the same directory as the code.

`unzip Alexnet-Feature-Extraction.zip`

`cd CarND-Alexnet-Feature-Extraction`

We add a line fc7 = tf.stop_gradient(fc7), Note tf.stop_gradient prevents the gradient from flowing backwards past this point, keeping the weights before and up to fc7 frozen. This also makes training faster, less work to do!

Train the AlexNet

`python train_feature_extraction.py`

Training AlexNet (even just the final layer!) can take a little while, so if you don't have a GPU, running on a subset of the data is a good alternative. As a point of reference one epoch over the training set takes roughly 53-55 seconds with a GTX 970.

## Comparisons of Inceptions(GoogLeNet), ReseNet, VGGnet

### Jianguo Zhang, April 22, 2017. Update in July, 27, 2017

![image5](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/GoogleNet.gif)

![image5](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/GoogleNet_2.jpg)

 We will use [Keras](https://keras.io/) to explore feature extraction with the VGG, Inception and ResNet architectures. The models you will use were trained for days or weeks on the ImageNet dataset. Thus, the weights encapsulate higher-level features learned from training on thousands of classes.


There are some notable differences from AlexNet program.

1. We're using two datasets. First, the German Traffic Sign dataset, and second, the Cifar10 dataset.

2. Bottleneck Features. Unless you have a very powerful GPU, running feature extraction on these models will take a significant amount of time, as you might have observed in the AlexNet lab. To make things easier we've precomputed bottleneck features for each (network, dataset) pair. This will allow you to experiment with feature extraction even on a modest CPU. You can think of bottleneck features as feature extraction but with caching. Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed through the network, we can cache and reuse the output.

3. Furthermore, we've limited each class in both training datasets to 100 examples. The idea here is to push feature extraction a bit further. It also greatly reduces the download size and speeds up training. The validation files remain the same.

The files are encoded as such:

* {network}_{dataset}_100_bottleneck_features_train.p
* {network}_{dataset}_bottleneck_features_validation.p

"network", in the above filenames, can be one of 'vgg', 'inception', or 'resnet'.

"dataset" can be either 'cifar10' or 'traffic'.

Download one of the bottleneck feature packs from [VGG Bottleneck Features 100](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip), [ResNet Bottleneck Features 100](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip) and [InceptionV3 Bottleneck Features 100](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip). VGG is the smallest so you might want to give that a shot first

Make sure the all the files are in the same directory as the code.

`unzip CarND-Transfer-Learning.zip`

`cd CarND-Transfer-Learning`



Here we define some command line flags like following, this avoids having to manually open and edit the file if we want to change the files we train and validate our model with.


* flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")

* flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")

Here's how you would run the file from the command line:

`python feature_extraction.py --training_file vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg_cifar10_bottleneck_features_validation.p`

After 50 epochs these are the results for each model on cifar-10 dataset:

**VGG**:

`Epoch 50/50`

`1000/1000 [==============================] - 0s - loss: 0.2418 - acc: 0.9540 - val_loss: 0.8759 - val_acc: 0.7235`

**Inception(GoogLeNet)**:

`Epoch 50/50`

`1000/1000 [==============================] - 0s - loss: 0.0887 - acc: 1.0000 - val_loss: 1.0428 - val_acc: 0.6556`

**ResNet**:

`Epoch 50/50`

`1000/1000 [==============================] - 0s - loss: 0.0790 - acc: 1.0000 - val_loss: 0.8005 - val_acc: 0.7347`

 Now do the same thing but with the German Traffic Sign dataset. The ImageNet dataset with 1000 classes had no traffic sign images. Will the high-level features learned still be transferable to such a different dataset?
 
 Staying with the VGG example:

 `python feature_extraction.py --training_file bottlenecks/vgg_traffic_100_bottleneck_features_train.p --validation_file bottlenecks/vgg_traffic_bottleneck_features_validation.p`

After 50 epochs these are the results for each model on the German Traffic Sign dataset:

**VGG**:

`Epoch 50/50`

`4300/4300 [==============================] - 0s - loss: 0.0873 - acc: 0.9958 - val_loss: 0.4368 - val_acc: 0.8666`

**Inception(GoogLeNet)**:

`Epoch 50/50`

`4300/4300 [==============================] - 0s - loss: 0.0276 - acc: 1.0000 - val_loss: 0.8378 - val_acc: 0.7519`

**ResNet**:

`Epoch 50/50`

`4300/4300 [==============================] - 0s - loss: 0.0332 - acc: 1.0000 - val_loss: 0.6146 - val_acc: 0.8108`
