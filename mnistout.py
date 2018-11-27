# -*- coding: utf-8 -*-  
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

img_train = mnist.test.images[108]
img_train = img_train

trainSet = np.zeros((28, 28))
#trainSet2 = np.zeros((1, 784))
trainSet2 =img_train
trainSet=trainSet2.reshape(28, 28)

print(trainSet)
pyplot.imshow(trainSet)
pyplot.show()
im.save("/home/zixuan/桌面/sample.png")

