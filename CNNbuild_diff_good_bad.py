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

BATCH_SIZE=10
PIC_LENGTH=1920
PIC_WIDTH=2560
PIC_DEPTH=3
TYPE=2
keep_prob=[0.85,0.9,0.95]

class PIC_CNN(object):
    def __init__(self):
        self.data_length=PIC_LENGTH
        self.data_width=PIC_WIDTH
        self.data_depth=PIC_DEPTH
        self.batch=BATCH_SIZE
        self.keep_prob=keep_prob
        self.get_train_file()
        self.train_len = len(self.train_map)
        self.split_data_file()
        self.build_graph()

    def get_train_file(self):
        Png_Data = GetPngData()
        self.train_map, self.label_map = Png_Data.get_train_data2()
            # arr=np.array([self.train_map,self.train_label_map])
        df = pd.Series(self.label_map)
        df.to_csv("stat.csv")
    def get_test_data(self):
        Png_Data = GetPngData()
        test_name,test_data = Png_Data.main_test()
        return test_name,test_data

    def read_train_img(self,train_map):
        train_data=[]
        #label_data=[]
        for i in range(len(train_map)):
            #pic_name=join(self.train_data_path,train_map[i])
            #print (pic_name)
            train_data.append(cv2.imread(train_map[i]))
        #for i in range(len(label_map)):
        #    label_data.append(label_map[i])
        #label_num=self.turn_label_to_num(label_data)
        #df_train=pd.DataFrame(train_data)
        return train_data

    def split_data_file(self):
        f_name = list(self.train_map.values())
        # print (f_name[0])
        label = list(self.label_map.values())
        self.train_data_map = []
        self.train_label_map = []
        self.test_data_map = []
        self.test_label_map = []
        rate = 0.9
        np.random.seed(100)
        shuf = np.random.permutation(np.arange(self.train_len))
        for i in range(int(len(shuf) * rate)):
            # print(shuf[i])
            # print(f_name[1666])
            self.train_data_map.append(f_name[shuf[i]])
            self.train_label_map.append(label[shuf[i]])
        for j in range(int(len(shuf) * rate), self.train_len):
            self.test_data_map.append(f_name[shuf[j]])
            self.test_label_map.append(label[shuf[j]])
        self.train_df = pd.DataFrame(np.array([self.train_data_map, self.train_label_map]).transpose(),
                                     columns=["path", "label"], index=self.train_data_map)

    def turn_label_to_num(self,label):

        label_dict={'无瑕疵样本':0, '不导电':1, '凸粉':1, '擦花':1,'桔皮' :1,'横条压凹':1,'涂层开裂':1, '漏底':1,'碰伤':1,'脏点':1, '起坑':1,'其他':1}
        df_label=pd.DataFrame(label)
        label_num=df_label[0].map(label_dict)
        return label_num
    def turn_num_to_label(self,num):
        label_dict={0:'norm', 1:'defect'}
        label=[label_dict[k] for k in num]
        return label

    def get_validate_data_from_file(self):
        data=self.read_train_img(self.test_data_map)
        label=self.turn_label_to_num(self.test_label_map)
        return np.asarray(data),np.asarray(label)
    def get_batch_data_from_file(self,rank):
        data_len=len(self.train_data_map)
        beg = (rank * self.batch) % data_len
        end = ((rank + 1) * self.batch) % data_len
        if beg < end:
            batch=self.train_data_map[beg:end]
            batch_data = self.read_train_img(batch)
            batch_label = self.train_label_map[beg:end]
            #print (batch)
        else:
            batch = self.train_data_map[beg:data_len]
            batch_label = self.train_label_map[beg:data_len]
            batch.extend(self.train_data_map[0:end])
            batch_label.extend(self.train_label_map[0:end])
            batch, batch_label = batch, batch_label
            batch_data=self.read_train_img(batch)
        label=self.turn_label_to_num(batch_label)
        #d1 = pd.DataFrame(batch)
        #d1.to_csv("test.csv")
        return np.asarray(batch_data), np.asarray(label)
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
            self.eval_logit=self.cnn_def_1(self.input_x)
            self.target_logit=tf.placeholder(tf.float32,[None,self.size_y],name="target_logits")

    def cnn_def_1(self,input_x,scope='CNN',collect='CNN'):
        with tf.name_scope(scope):
            with tf.name_scope('layer1'):
                #[-1,1920,2560,3] -> [-1,120,160,3] -> [-1,60,80,3]
                lay_1_channel = 3
                w_conv1 = tf.Variable(tf.truncated_normal([24, 24, 3, lay_1_channel]), name='w_conv1')
                b_conv1 = tf.Variable(tf.constant(0.1, shape=[lay_1_channel]), name='b_conv1')
                v_conv1 = tf.nn.conv2d(input_x, w_conv1, strides=[1, 16, 16, 1], padding='SAME', name='v_conv1')
                h_conv1 = tf.nn.relu( v_conv1 + b_conv1, name='h_conv1')
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                                         name='pool1')
                h_out1 = h_pool1
            with tf.name_scope("layer_2"):
                #[-1,60,80,3] -> [-1,20,27,10] -> [-1,10,14,10]
                lay_2_channel=10
                w_conv2 = tf.Variable(tf.truncated_normal([7, 7, lay_1_channel, lay_2_channel]), name='w_conv2')
                b_conv2 = tf.Variable(tf.constant(0.01, shape=[lay_2_channel]), name='b_conv2')
                v_conv2 = tf.nn.conv2d(h_out1, w_conv2, strides=[1, 3, 3, 1], padding='SAME', name='v_conv2')
                h_conv2 = tf.nn.relu(v_conv2 + b_conv2, name='h_conv2')
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
                h_out2 = h_pool2
            with tf.name_scope("layer_3"):
                #[-1,10,14,10] -> [-1,5,7,5] -> [-1,5,7,5]
                lay_3_channel=5
                w_conv3 = tf.Variable(tf.truncated_normal([5,5,lay_2_channel, lay_3_channel]), name='w_conv3')
                b_conv3 = tf.Variable(tf.constant(0.01, shape=[lay_3_channel]), name='b_conv3')
                v_conv3 = tf.nn.conv2d(h_out2, w_conv3, strides=[1, 2, 2, 1], padding='SAME', name='v_conv3')
                h_conv3 = tf.nn.relu(v_conv3 + b_conv3, name='h_conv3')
                h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
                h_out3 = h_pool3
            with tf.name_scope("fully_connected_layer1"):
                h_flat1 = tf.reshape(h_out3, [-1, 5 * 7 * lay_3_channel])
                w_fc1=tf.Variable(tf.truncated_normal([5*7*lay_3_channel,48],stddev=0.1),name='wfc1')
                b_fc1=tf.Variable(tf.constant(0.05,shape=[48]),name='b_fc1')
                v_fc1=tf.matmul(h_flat1,w_fc1,name='v_fc1')
                h_fc1=tf.nn.relu(v_fc1+b_fc1,name='h_fc1')
                h_drop1=tf.nn.dropout(h_fc1,self.keep_prob[0],name='h_drop1')
            with tf.name_scope("fully_connected_layer2"):
                w_fc2=tf.Variable(tf.truncated_normal([48,16],stddev=0.1),name='wfc2')
                b_fc2=tf.Variable(tf.constant(0.05,shape=[16]),name='b_fc2')
                v_fc2=tf.matmul(h_drop1,w_fc2,name='v_fc2')
                h_fc2=tf.nn.relu6(v_fc2+b_fc2,name='h_fc2')
                h_drop2=tf.nn.dropout(h_fc2,self.keep_prob[1],name='h_drop2')
            with tf.name_scope("fully_connected_layer3"):
                w_fc3 = tf.Variable(tf.truncated_normal([16,self.size_y],stddev=0.1), name='wfc3')
                b_fc3 = tf.Variable(tf.constant(0.05, shape=[self.size_y]), name='b_fc3')
                v_fc3 = tf.matmul(h_drop2, w_fc3, name='v_fc3')
                h_fc3=tf.add(v_fc3,b_fc3,name='h_fc3')
                #out=tf.nn.sigmoid(h_fc3,name='out')
                return h_fc3
    def train_para(self):
        with self.graph.as_default():
            with tf.name_scope("loss"):
                self.label=self.one_hot_label(self.input_y)
                #print(self.input_y.get_shape())
                #print(self.input_x.get_shape())
                #print(self.eval_logit.get_shape())
                self.cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.eval_logit,name='cross_entropy')

                self.cross_entropy_mean=tf.reduce_mean(self.cross_entropy,name='cross_entropy_mean')
            with tf.name_scope("train"):
                self._lr=tf.Variable(0.0,trainable=False)
                self._new_lr=tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
                self._lr_update=tf.assign(self._lr,self._new_lr)
                self.global_step=tf.Variable(0,name='global_step',trainable=False)
                self.optimizer=tf.train.MomentumOptimizer(self._lr,momentum=0.9)
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy_mean, tvars), 5)
                self.train_step = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
            with tf.name_scope("test"):
                self.prediction=tf.argmax(self.eval_logit,1,name="prediction")
                #self.target_prediction=tf.nn.softmax(self.target_logit,name="target_prediction")
                self.correct_prediction=tf.equal(tf.cast(self.input_y,tf.int32),tf.cast(self.prediction,tf.int32),name="correct_prediction")
                self.accuracy=tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32),name="accuracy")
            with tf.name_scope("predict"):
                self.predict=tf.argmax(self.eval_logit,1,name="predict_label")
    def train(self):
        with tf.Session(graph=self.graph,config=
                        tf.ConfigProto(inter_op_parallelism_threads=3,
                                       intra_op_parallelism_threads=15)) as sess:
            sess.run(tf.global_variables_initializer());
            lr_init = 1e-3
            lr_end = 1e-4
            i = 0
            run_step=0
            train_times = 20
            rank=0
            acc=0
            #print (self.train_len)
            #train, train_label, test, test_label = self.split_data()
            x_test,y_test=self.get_validate_data_from_file()
            #print(x_test.shape)
            name,predict_data=self.get_test_data()
            while run_step<1000 and acc<=0.9:
                if i <  500:
                    cur_lr = lr_init
                else:
                    cur_lr = lr_end
                sess.run(self._lr_update,feed_dict={self._new_lr: cur_lr})
                #x=train
                #y=train_label

                x,y=self.get_batch_data_from_file(rank)

                #print(y)
                rank=rank+1
                #print (x.shape)
                #print (y)
                #print (test)
                #print(y)
                #x_test=test[0]
                #y_test=test[1]
                sess.run(self.train_step,feed_dict={self.input_x: x,self.input_y: y})
                loss=sess.run(self.cross_entropy_mean,feed_dict={self.input_x: x, self.input_y:y})
                train_acc=sess.run(self.accuracy,feed_dict={self.input_x:x,self.input_y:y})
                test_loss=sess.run(self.cross_entropy_mean,feed_dict={self.input_x: x, self.input_y:y})
                test_acc = sess.run(self.accuracy, feed_dict={self.input_x: x_test, self.input_y: y_test})
                if run_step % 50==0:
                    print ("run_step: %g, train_loss: %g, test_loss: %g, train_acc: %g, test_acc: %g" % (run_step,loss,test_loss,train_acc,test_acc))
                run_step=run_step+1;
                i=i+1;
            predict_label=sess.run(self.predict,feed_dict={self.input_x: predict_data})
            label=self.turn_num_to_label(predict_label)
            self.write_submit_file(name,label)
            #print (self.turn_num_to_label(predict_label))
        return 0
    def write_submit_file(self,jpg,label):
        mat=np.array([jpg,label])
        df=pd.DataFrame(mat.transpose(),columns=['jpg','prediction']);
        df_sorted=df.sort_values('jpg')
        #filename="my_submit.csv"
        df_sorted.to_csv("my_submit_0.csv",index=False,header=False)


def main_proc():
    gd=PIC_CNN();
    gd.train_para()
    gd.train()

if __name__=="__main__":
    main_proc()