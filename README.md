# Transfer-Learning
### Jianguo Zhang, April 17, 2017

Most of the time you won't want to train a whole convolutional network yourself. Modern ConvNets training on huge datasets like ImageNet take weeks on multiple GPUs. Instead, most people use a pretrained network either as a fixed feature extractor, or as an initial network to fine tune. In this notebook, you'll be using [VGGNet](https://arxiv.org/pdf/1409.1556.pdf) trained on the [ImageNet dataset](http://www.image-net.org/) as a feature extractor. Below is a diagram of the VGGNet architecture.

<img src="assets/cnnarchitecture.jpg" width=700px>

VGGNet is great because it's simple and has great performance, coming in second in the ImageNet competition. The idea here is that we keep all the convolutional layers, but replace the final fully connected layers with our own classifier. This way we can use VGGNet as a feature extractor for our images then easily train a simple classifier on top of that. What we'll do is take the first fully connected layer with 4096 units, including thresholding with ReLUs. We can use those values as a code for each image, then build a classifier on top of those codes.


In practice, we won't typically be training our own huge networks since it may take several days. 
There are multiple models out there that have been trained for weeks on huge datasets like ImageNet.
In this project, we use one of these pretrained networks, [VGGNet](https://github.com/machrisaa/tensorflow-vgg), to classify images of flowers.

Make sure you clone this repository into the transfer-learning directory.

`cd  transfer-learning`

`git clone https://github.com/machrisaa/tensorflow-vgg.git tensorflow_vgg`

## Installation

The `requirements.txt` describing the minimal dependencies required to run this program.

If you have most packages, you can just install Additional Packages

`pip install tqdm`

`conda install scikit-image`

### pip

To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`.


