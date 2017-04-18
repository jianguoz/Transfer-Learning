# Transfer-Learning
### Jianguo Zhang, April 17, 2017

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


