# -*- coding: utf-8 -*-  

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf

sess = tf.InteractiveSession()


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#cross_entropy = tf.reduce_mean(
    #tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

saver = tf.train.Saver(max_to_keep=1)  # defaults to saving all variables

sess.run(tf.initialize_all_variables())
#sess.run(tf.global_variables_initializer())


for i in range(10000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))

  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

saver.save(sess, '/home/zixuan/桌面/form/modeltest.ckpt')  #保存模型参数，注意把这里改为自己的路径
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images[:100], y_: mnist.test.labels[:100], keep_prob: 1.0}))
#------------------------------------------------------------

from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
def imageprepare():
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    file_name='/home/zixuan/桌面/test/sharpoutline9.png'#导入自己的图片地址
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name).convert('L')


    im.save("/home/zixuan/桌面/sample.png")
    plt.imshow(im)
    plt.show()
    tv = list(im.getdata()) #get pixel values
    print(tv)
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    #tva = [ (255-x)*1/255.0 for x in tv] 
    tva = [ round((x/255.000)*1.000,8) for x in tv] 
    print(tva)
    return tva

import numpy
result=imageprepare()
ndresult = numpy.array(result)

prediction=tf.argmax(y_conv,1)

#a = prediction.eval(feed_dict={
    #x: [mnist.test.images[95]], y_: numpy.array([[1,0,0,0,0,0,0,0,0,0]]), keep_prob: 1.0})
a = prediction.eval(feed_dict={
    x: [ndresult], y_: numpy.array([[1,0,0,0,0,0,0,0,0,0]]), keep_prob: 1.0})
print('a:')
print(a)




#print("my accuracy %g"%accuracy.eval(feed_dict={
    #x: numpy.array([result]), y_: numpy.array([[0,0,0,0,0,0,0,1,0,0]]), keep_prob: 1.0}))
#print(mnist.test.images[1])
#print(type(mnist.test.images[1]))
#print("test accuracy %g"%accuracy.eval(feed_dict={
#    x: mnist.test.images[:500], y_: mnist.test.labels[:500], keep_prob: 1.0}))
#print(result)
#a = prediction.eval(feed_dict={
#    x: numpy.array([result]), y_: numpy.array([[1,0,0,0,0,0,0,0,0,0]]), keep_prob: 1.0})
print('recognize result:')
#print(a)

#a = y_conv.eval(feed_dict={
#    x: numpy.array([result]), y_: numpy.array([[0,0,0,1,0,0,0,0,0,0]]), keep_prob: 1.0})
#print(a)
#print(predint[0])
#print(predint)


