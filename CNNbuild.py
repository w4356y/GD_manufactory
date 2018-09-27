#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from extract import *
from os.path import join
import tensorflow as tf

cur_dir = os.path.dirname(os.path.abspath(__file__))
#print (cur_dir)
os.chdir(cur_dir)

BATCH_SIZE=121
PIC_LENGTH=1920
PIC_WIDTH=2560
PIC_DEPTH=3
TYPE=12
keep_prob=[1,1,1]
class GD_CNN(object):
    def __init__(self):
        self.data_length=PIC_LENGTH
        self.data_width=PIC_WIDTH
        self.data_depth=PIC_DEPTH
        self.batch=BATCH_SIZE
        self.keep_prob=keep_prob
        self.build_graph()
    def get_train_data(self):
        Png_Data = GetPngData()
        train_data,train_label=Png_Data.main_train()
        self.data_len=len(train_label)
        return [train_data,train_label]

    def get_test_data(self):
        Png_Data = GetPngData()
        test_data = Png_Data.main_test()
        return test_data
    def split_data(self):
        data,label=self.get_train_data();
        #print (label)
        train_data=[]
        train_label=[]
        test_data=[]
        test_label=[]
        rate=0.9
        shuf=np.random.permutation(np.arange(self.data_len))
        for i in range(int(len(shuf)*rate)):
            train_data.append(data[shuf[i]])
            train_label.append(label[shuf[i]])
        for j in range(int(len(shuf)*rate),self.data_len):
            test_data.append(data[shuf[j]])
            test_label.append(label[shuf[j]])
        return [train_data,train_label,test_data,test_label]

    def get_batch_data(self,data,label,rank):
        data_len=len(data)
        beg=(rank*self.batch) % data_len
        end=((rank+1)*self.batch) % data_len
        if beg<end:
            batch_data=data[beg:end]
            batch_label=label[beg:end]
        else:
            batch_data=data[beg:data_len]
            batch_label=label[beg:data_len]
            batch_data.extend(data[0:end])
            batch_label.extend(label[0:end])
            batch_data,batch_label=batch_data,batch_label
        return np.asarray(batch_data), np.asarray(batch_label)
    def get_batch_test_data(self,data,label,size=100):
        data_len=len(data)
        batch_test=data[0:size]
        batch_label=label[0:size]
        return np.asarray(batch_test),np.asarray(batch_label)
    def one_hot_label(self,input_y):
        label=tf.one_hot(input_y,self.size_y,on_value=1,off_value=None,axis=1)
        return label

    def build_graph(self):
        self.graph=tf.Graph()
        with self.graph.as_default():
            self.input_x=tf.placeholder(tf.float32,[None,self.data_length,self.data_width,3],name='input_x')
            self.input_y=tf.placeholder(tf.int32,[None],name='input_y')
            #self.test_x = tf.placeholder(tf.float32, [None, self.data_length, self.data_width, 3], name='test_x')
            #self.test_y = tf.placeholder(tf.int32, [None], name='test_y')
            self.size_y=TYPE
            self.pro_holder=tf.placeholder(tf.float32,name='keep_prob')
            self.eval_logit=self.cnn_def(self.input_x,self.input_y,self.size_y)
            self.target_logit=tf.placeholder(tf.float32,[None,self.size_y],name="target_logits")

    def cnn_def(self,input_x,input_y,size_y,scope='CNN',collect='CNN'):
        with tf.name_scope(scope):
            with tf.name_scope('layer1'):
                #[-1,1920,2560,3] -> [-1,1920,2560,3]
                lay_1_channel=3
                w_conv1=tf.Variable(tf.truncated_normal([3,3,3,lay_1_channel]),name='w_conv1')
                b_conv1=tf.Variable(tf.constant(0.1,shape=[lay_1_channel]),name='b_conv1')
                v_conv1=tf.nn.conv2d(input_x,w_conv1,strides=[1,1,1,1],padding='SAME',name='v_conv1')
                h_conv1=tf.nn.relu(v_conv1+b_conv1,name='h_conv1')
                h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME',name='pool1')
                h_out1=h_pool1

            with tf.name_scope("layer_2"):
                #[-1,1920,2560,128] -> [-1,960,1280,64] -> [-1,480,640,64]
                lay_2_channel=3
                w_conv2 = tf.Variable(tf.truncated_normal([5, 5, lay_1_channel, lay_2_channel]), name='w_conv2')
                b_conv2 = tf.Variable(tf.constant(0.01, shape=[lay_2_channel]), name='b_conv2')
                v_conv2 = tf.nn.conv2d(h_out1, w_conv2, strides=[1, 2, 2, 1], padding='SAME', name='v_conv2')
                h_conv2 = tf.nn.relu(v_conv2 + b_conv2, name='h_conv2')
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
                h_out2 = h_pool2

            with tf.name_scope("layer_3"):
                #[-1,480,640,64] -> [-1,120,160,64] -> [-1,60,80,64]
                lay_3_channel=3
                w_conv3 = tf.Variable(tf.truncated_normal([3, 3, lay_2_channel, lay_3_channel]), name='w_conv3')
                b_conv3 = tf.Variable(tf.constant(0.01, shape=[lay_3_channel]), name='b_conv3')
                v_conv3 = tf.nn.conv2d(h_out2, w_conv3, strides=[1, 4, 4, 1], padding='SAME', name='v_conv3')
                h_conv3 = tf.nn.relu(v_conv3 + b_conv3, name='h_conv3')
                h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
                h_out3 = h_pool3

            with tf.name_scope("layer_4"):
                #[-1,60,80,64] -> [-1,30,40,48] -> [-1,15,20,48]
                lay_4_channel = 3
                w_conv4 = tf.Variable(tf.truncated_normal([5, 5, lay_3_channel, lay_4_channel]), name='w_conv4')
                b_conv4 = tf.Variable(tf.constant(0.01, shape=[lay_4_channel]), name='b_conv4')
                v_conv4 = tf.nn.conv2d(h_out3, w_conv4, strides=[1, 2, 2, 1], padding='SAME', name='v_conv4')
                h_conv4 = tf.nn.relu(v_conv4 + b_conv4, name='h_conv4')
                h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                         name='pool4')
                h_out4= h_pool4

            with tf.name_scope("layer_5"):
                #[-1,15,20,48] -> [-1,5,7,24] -> [-1,5,7,24]
                lay_5_channel = 3
                w_conv5 = tf.Variable(tf.truncated_normal([4, 4, lay_4_channel, lay_5_channel]), name='w_conv5')
                b_conv5 = tf.Variable(tf.constant(0.01, shape=[lay_5_channel]), name='b_conv5')
                v_conv5 = tf.nn.conv2d(h_out4, w_conv5, strides=[1, 3, 3, 1], padding='SAME', name='v_conv5')
                h_conv5 = tf.nn.relu(v_conv5 + b_conv5, name='h_conv5')
                h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                         name='pool5')
                h_out5= h_pool5

            with tf.name_scope("fully_connected_layer1"):
                h_flat1 = tf.reshape(h_out5, [-1, 5 * 7 * lay_5_channel])
                w_fc1=tf.Variable(tf.truncated_normal([5*7*lay_5_channel,2048],stddev=0.1),name='wfc1')
                b_fc1=tf.Variable(tf.constant(0.05,shape=[2048]),name='b_fc1')
                v_fc1=tf.matmul(h_flat1,w_fc1,name='v_fc1')
                h_fc1=tf.nn.relu(v_fc1+b_fc1,name='h_fc1')
                h_drop1=tf.nn.dropout(h_fc1,self.keep_prob[0],name='h_drop1')

            with tf.name_scope("fully_connected_layer2"):
                w_fc2=tf.Variable(tf.truncated_normal([2048,128],stddev=0.1),name='wfc2')
                b_fc2=tf.Variable(tf.constant(0.05,shape=[128]),name='b_fc2')
                v_fc2=tf.matmul(h_drop1,w_fc2,name='v_fc2')
                h_fc2=tf.nn.relu(v_fc2+b_fc2,name='h_fc2')
                h_drop2=tf.nn.dropout(h_fc2,self.keep_prob[1],name='h_drop2')

            with tf.name_scope("fully_connected_layer3"):
                w_fc3=tf.Variable(tf.truncated_normal([128,32],stddev=0.1),name='wfc3')
                b_fc3=tf.Variable(tf.constant(0.05,shape=[32]),name='b_fc3')
                v_fc3=tf.matmul(h_drop2,w_fc3,name='v_fc3')
                h_fc3 = tf.nn.relu(v_fc3 + b_fc3, name='h_fc3')
                h_drop3 = tf.nn.dropout(h_fc3, self.keep_prob[2], name='h_drop3')

            with tf.name_scope("fully_connected_layer4"):
                w_fc4 = tf.Variable(tf.truncated_normal([32,self.size_y],stddev=0.1), name='wfc4')
                b_fc4 = tf.Variable(tf.constant(0.05, shape=[self.size_y]), name='b_fc4')
                v_fc4 = tf.matmul(h_drop3, w_fc4, name='v_fc4')
                h_fc4=tf.add(v_fc4,b_fc4,name='h_fc4')
                return h_fc4

    #def buidCNN(self):
    #    self.graph=tf.Graph()
    #    with self.graph.as_default():

    def train_para(self):
        with self.graph.as_default():
            with tf.name_scope("loss"):
                self.label=self.one_hot_label(self.input_y)
                print(self.input_y.get_shape())
                print(self.input_x.get_shape())
                #print(self.eval_logit.get_shape())
                self.cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.eval_logit,name='cross_entropy')

                self.cross_entropy_mean=tf.reduce_mean(self.cross_entropy,name='cross_entropy_mean')
            with tf.name_scope("train"):
                self._lr=tf.Variable(0.0,trainable=False)
                self._new_lr=tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
                self._lr_update=tf.assign(self._lr,self._new_lr)
                self.global_step=tf.Variable(0,name='global_step',trainable=False)
                self.optimizer=tf.train.AdamOptimizer(self._lr)
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy_mean, tvars), 5)
                self.train_step = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            with tf.name_scope("test"):
                self.prediction=tf.argmax(self.eval_logit,1,name="prediction")
                #self.target_prediction=tf.nn.softmax(self.target_logit,name="target_prediction")
                self.correct_prediction=tf.equal(tf.cast(self.input_y,tf.int32),tf.cast(self.prediction,tf.int32),name="correct_prediction")
                self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32),name="accuracy")

    def train(self):
        with tf.Session(graph=self.graph,config=
                        tf.ConfigProto(inter_op_parallelism_threads=1,
                                       intra_op_parallelism_threads=2)) as sess:
            sess.run(tf.global_variables_initializer());
            lr_init = 1e-4
            lr_end = 1e-5
            i = 0
            run_step=0
            train_times = 20
            rank=0
            train, train_label, test, test_label = self.split_data()
            while run_step<10:
                if i < train_times * 1 / 4:
                    cur_lr = lr_init
                else:
                    cur_lr = lr_end
                sess.run(self._lr_update,feed_dict={self._new_lr: cur_lr})
                #x=train
                #y=train_label
                x,y=self.get_batch_data(train,train_label,rank)
                rank=rank+1
                x_test,y_test=self.get_batch_test_data(test,test_label)
                print (y)
                #print (test)
                #print(y)
                #x_test=test[0]
                #y_test=test[1]
                sess.run(self.train_step,feed_dict={self.input_x: x,self.input_y: y})
                loss=sess.run(self.cross_entropy_mean,feed_dict={self.input_x: x, self.input_y:y})
                acc = sess.run(self.accuracy, feed_dict={self.input_x: np.asarray(test), self.input_y: np.asarray(test_label)})
                print ("loss: %g, acc: %g" % (loss,acc))
                run_step=run_step+1;
        return 0

def main_proc():
    gd=GD_CNN();
    gd.train_para()
    gd.train()

if __name__=="__main__":
    main_proc()



