# -*- coding: utf-8 -*-  
from flask import Flask ,redirect, url_for
from redis import Redis, RedisError
from flask import request
from flask import render_template
import re
from cStringIO import StringIO
from PIL import Image, ImageFilter
import os,base64
import socket
#-----------
import numpy
import tensorflow as tf

import logging
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement

import datetime

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)

# Prepare Cassandra

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

KEYSPACE = "keyspace1"
session = 0

def createKeySpace():
    global session
    cluster = Cluster(contact_points=['cassandra'],port=9042)
    session = cluster.connect()
    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2'  }
            """ % KEYSPACE)
        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)
        log.info("creating table...")
        session.execute("""
            CREATE TABLE data (
		id int,
		imgname text,
		img text,
		time text,
		result text,
                PRIMARY KEY (id)
            )
            """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)
idcount = 1
createKeySpace();
# Prepare Cassandra end

@app.route('/')
def hello_world():
    global idcount
    global session
    isrc = request.args.get('pic')

    print(isrc)
    if(isrc != None):
	#图片转为矩阵
	isrctrans = isrc.replace(' ','+')  #trans" " back to "+"
	img = base64_to_image(isrctrans)
	#plt.imshow(img)
	#plt.show()
	tv = list(img.getdata())
	tva = [ round((x/255.000)*1.000,8) for x in tv] 
	#识别图片
	result = recnub(tva)
	#存入Cassandra
	imgname  = "testpic"+str(idcount)
	img
	session.execute("""INSERT INTO """+KEYSPACE+"""."""+"""data (id,imgname, img, time, result) VALUES ("""+str(idcount)+""",'"""+ imgname+"""','"""+isrctrans+"""','"""+str(datetime.datetime.now())+"""','"""+ str(result)+"""');""")
		#每次存好显示数据表
	session.set_keyspace(KEYSPACE)
	ctable = session.execute("""SELECT * FROM data;""")
	print(ctable)
	idcount = idcount+1;
    	return render_template('jSig.html', message=result,picsrc = isrctrans)
    name = "test"
    return render_template('jSig.html', message="none")

def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    binary_data = base64.b64decode(base64_data)
    img_data = StringIO(binary_data)
    img = Image.open(img_data).convert('L')
    if image_path:
        img.save(image_path)
    return img




def recnub(data):#(784维数组)
#model
	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
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
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
	init_op = tf.initialize_all_variables()
	#model
	ndresult = numpy.array(data)	
	prediction=tf.argmax(y_conv,1)
	#saver
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init_op)
		saver.restore(sess, "./form/modeltest.ckpt")#这里使用了之前保存的模型参数
	#saver
		print(ndresult)
		a = prediction.eval(feed_dict={x: [ndresult], keep_prob: 1.0})
		print('rerereresult:')
		print(a)
		return a


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   










@app.route("/file",methods=['POST','GET'])
def filein():

#    try:
#        visits = redis.incr("counter")
#    except RedisError:
#        visits = "<i>cannot connect to Redis, counter disabled</i>"

#    try:
#        file = request.form['file']
#    except:
#        file = None
#    if file:  #getpng start rec
#	text= "we get the file:"+file.filename
#	KEYSPACE = "keyspace1"
	#inserttest();
    #pic = request.form['picdata']
    pic = request.args.get('imgdata')

    #return render_template('showresult.html', message="result")
    return redirect(url_for('hello_world',pic = pic,result = 15))




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
