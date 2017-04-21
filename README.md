# Transfer-Learning

## VGGNet

**Jianguo Zhang, April 17, 2017**

Most of the time we won't want to train a whole convolutional network yourself. Modern ConvNets training on huge datasets like ImageNet take weeks on multiple GPUs. Instead, most people use a pretrained network either as a fixed feature extractor, or as an initial network to fine tune. In this notebook, we'll be using [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) trained on the [ImageNet dataset](http://www.image-net.org/) as a feature extractor. Below is a diagram of the VGGNet architecture.


![image1](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/assets/cnnarchitecture.jpg)

VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully connected layers with our own classifier. So what we want are the values of the first fully connected layer, after being ReLUd (self.relu6). This way we can use VGGNet as a feature extractor for our images then easily train a simple classifier on top of that. What we'll do is take the first fully connected layer with 4096 units, including thresholding with ReLUs. We can use those values as a code for each image, then build a classifier on top of those codes.

 There are more informations about transfer learning from [the CS231n course notes](http://cs231n.github.io/transfer-learning/#tf).


Make sure you clone this repository into the transfer-learning directory.

`cd  transfer-learning`

`git clone https://github.com/machrisaa/tensorflow-vgg.git tensorflow_vgg`

### Installation

The program using `TensorFlow==1.0.0` and `python==3.6.0 `. The `requirements.txt` describing the minimal dependencies required to run this program. 

If you have most packages, you can just install Additional Packages

`pip install tqdm`

`conda install scikit-image`

### pip

To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`.


## AlexNet

**Jianguo Zhang, April 19, 2017**

AlexNet is a popular base network for transfer learning because its structure is relatively straightforward, it's not too big, and it performs well empirically.

![image](https://github.com/JianguoZhang1994/Transfer-Learning/blob/master/AlexNet_image.png)


In this program, we extract AlexNet's features and use them to classify images from the [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), which includes 43 classes. The orignal AlexNet classifies for 1000 classes.

Download the [train data](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a829f_train/train.p) and [AlexNet weights](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npyhttps://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d880c_bvlc-alexnet/bvlc-alexnet.npy)


Make sure the all the files are in the same directory as the code.

`unzip Alexnet-Feature-Extraction.zip`

`cd CarND-Alexnet-Feature-Extraction`

We add a line fc7 = tf.stop_gradient(fc7), Note `tf.stop_gradient` prevents the gradient from flowing backwards past this point, keeping the weights before and up to `fc7` frozen. This also makes training faster, less work to do!

Train the AlexNet

`python train_feature_extraction.py`

Training AlexNet (even just the final layer!) can take a little while, so if you don't have a GPU, running on a subset of the data is a good alternative. As a point of reference one epoch over the training set takes roughly 53-55 seconds with a GTX 970.



