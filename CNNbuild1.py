#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from extract import *
from data_aug import *
from os.path import join
import tensorflow as tf

cur_dir = os.path.dirname(os.path.abspath(__file__))
#print (cur_dir)
os.chdir(cur_dir)

BATCH_SIZE=90
PIC_LENGTH=640
###1920 orig
PIC_WIDTH=640
###2560 orig
PIC_DEPTH=3
TYPE=12
keep_prob=[0.85,0.9,0.95]
class GD_CNN(object):
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
        self.size_y=TYPE
    def get_train_file(self):
        Png_Data=GetPngData()
        self.train_map,self.label_map=Png_Data.get_train_data2()
        #arr=np.array([self.train_map,self.train_label_map])
        df=pd.Series(self.label_map)
        df.to_csv("stat.csv")

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
        f_name=list(self.train_map.values())
        #print (f_name[0])
        label=list(self.label_map.values())
        self.train_data_map = []
        self.train_label_map = []
        self.test_data_map = []
        self.test_label_map = []
        rate = 0.9
        shuf = np.random.permutation(np.arange(self.train_len))
        for i in range(int(len(shuf) * rate)):
            #print(shuf[i])
            #print(f_name[1666])
            self.train_data_map.append(f_name[shuf[i]])
            self.train_label_map.append(label[shuf[i]])
        for j in range(int(len(shuf) * rate), self.train_len):
            self.test_data_map.append(f_name[shuf[j]])
            self.test_label_map.append(label[shuf[j]])
        self.train_df=pd.DataFrame(np.array([self.train_data_map,self.train_label_map]).transpose(),columns=["path","label"],index=self.train_data_map)
        #return [train_data_map, train_label_map, test_data_map, test_label_map]
    def get_validate_data_from_file(self):
        #data1=[]
        #label1=[]
        data=self.read_train_img(self.test_data_map)
        label=self.turn_label_to_num(self.test_label_map)
        #for i in range(len(data)):
        #    if label[i]!=0:
        #        data1.append(data[i])
        #        label1.append(label[i])
        return np.asarray(data),np.asarray(label)

    def get_even_distrib_batch_data(self,rank):
        batch=[]
        #label=[]
        for cat in self.train_df.label.unique():
            #if cat !='无瑕疵样本':
            set=list(self.train_df.path[self.train_df.label==cat])
            if len(set)>10:
                beg=(rank*10) % len(set)
                end= ((rank+1)*10) % len(set)
                if beg<end:
                    batch.extend(set[beg:end])
                else:
                    batch.extend(set[beg:len(set)])
                    batch.extend(set[0:end])
            else:
                sampling=np.random.choice(set,10,replace=True)
                batch.extend(sampling)
        #print (batch)
        batch_data=self.read_train_img(batch)
        batch_label=list(self.train_df.loc[batch,'label'])
        label = self.turn_label_to_num(batch_label)
        return np.asarray(batch_data),np.asarray(label)


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

    def get_train_data(self):
        Png_Data = GetPngData()
        train_data,train_label=Png_Data.main_train()
        self.data_len=len(train_label)
        return [train_data,train_label]

    def get_test_data(self):
        Png_Data = GetPngData()
        test_name,test_data = Png_Data.main_test()
        return test_name,test_data
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

    def turn_label_to_num(self,label):

        label_dict={'无瑕疵样本':0, '不导电':1, '凸粉':2, '擦花':3,'桔皮' :4,'横条压凹':5,'涂层开裂':6, '漏底':7,'碰伤':8,'脏点':9, '起坑':10,'其他':11}
        df_label=pd.DataFrame(label)
        label_num=df_label[0].map(label_dict)

        return label_num
    def turn_num_to_label(self,num):
        label_dict={0:'norm', 1:'defect1', 2:'defect8', 3:'defect2',4: 'defect4',5:'defect3',6:'defect9',7:'defect5',8:'defect6',9:'defect10', 10:'defect7',11:'defect11'}
        label=[label_dict[k] for k in num]
        return label

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
            #self.size=tf.placeholder(tf.int32,name="size")
            #self.size = tf.Variable(0, name='size', trainable=False)
            #self.test_x = tf.placeholder(tf.float32, [None, self.data_length, self.data_width, 3], name='test_x')
            #self.test_y = tf.placeholder(tf.int32, [None], name='test_y')
            self.size_y=TYPE
            self.pro_holder=tf.placeholder(tf.float32,name='keep_prob')
            self.eval_logit=self.cnn_def_1(self.input_x)
            self.target_logit=tf.placeholder(tf.float32,[None,self.size_y],name="target_logits")
    def cnn_def_1(self,input_x,scope='CNN',collect='CNN'):
        with tf.name_scope(scope):
            """
            with tf.name_scope('data_augmentation'):
                input = tf.cast(input_x, tf.float32)
                #distorted_image = tf.random_crop(input, [self.batch,640, 640, 3], seed=666)
                #print (input_x.shape)
                distorted_image=input
                flip = lambda x: tf.image.random_flip_left_right(x)
                flip_imgs = tf.map_fn(flip, distorted_image)
                #distorted_image = tf.image.random_flip_left_right(image=distorted_image)
                distorted_image = tf.image.random_brightness(flip_imgs,
                                                             max_delta=63)
                distorted_image = tf.image.random_contrast(distorted_image,
                                                           lower=0.2, upper=1.8)
                norm= lambda x: tf.image.per_image_standardization(x)
                float_image=tf.map_fn(norm,distorted_image)
                #float_image = tf.image.per_image_standardization(distorted_image)
                float_image.set_shape([self.batch,1920, 2560, 3])
            """
            with tf.name_scope('layer1'):
                #[-1,640,640,3] -> [-1,160,160,3] -> [-1,80,80,3]
                lay_1_channel = 3
                w_conv1 = tf.Variable(tf.truncated_normal([5, 5, 3, lay_1_channel]), name='w_conv1')
                b_conv1 = tf.Variable(tf.constant(0.01, shape=[lay_1_channel]), name='b_conv1')
                v_conv1 = tf.nn.conv2d(input_x, w_conv1, strides=[1, 4, 4, 1], padding='SAME', name='v_conv1')
                h_conv1 = tf.nn.relu(v_conv1 + b_conv1, name='h_conv1')
                h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME',
                                         name='pool1')

                h_out1 = h_pool1
            with tf.name_scope("layer_2"):
                #[-1,80,80,3] -> [-1,27,27,3] -> [-1,14,14,32]
                lay_2_channel=32
                w_conv2 = tf.Variable(tf.truncated_normal([5, 5, lay_1_channel, lay_2_channel]), name='w_conv2')
                b_conv2 = tf.Variable(tf.constant(0.01, shape=[lay_2_channel]), name='b_conv2')
                v_conv2 = tf.nn.conv2d(h_out1, w_conv2, strides=[1,3, 3, 1], padding='SAME', name='v_conv2')
                h_conv2 = tf.nn.relu(v_conv2 + b_conv2, name='h_conv2')
                h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
                h_out2 = h_pool2
            with tf.name_scope("layer_3"):
                #[-1,14,14,32] -> [-1,7,7,24] -> [-1,7,7,24]
                lay_3_channel=24
                w_conv3 = tf.Variable(tf.truncated_normal([3,3,lay_2_channel, lay_3_channel]), name='w_conv3')
                b_conv3 = tf.Variable(tf.constant(0.01, shape=[lay_3_channel]), name='b_conv3')
                v_conv3 = tf.nn.conv2d(h_out2, w_conv3, strides=[1, 2, 2, 1], padding='SAME', name='v_conv3')
                h_conv3 = tf.nn.relu(v_conv3 + b_conv3, name='h_conv3')
                h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pool3')
                h_out3 = h_pool3
            with tf.name_scope("fully_connected_layer1"):
                h_flat1 = tf.reshape(h_out3, [-1, 7 * 7 * lay_3_channel])
                w_fc1=tf.Variable(tf.truncated_normal([7*7*lay_3_channel,1024],stddev=0.1),name='wfc1')
                b_fc1=tf.Variable(tf.constant(0.01,shape=[1024]),name='b_fc1')
                v_fc1=tf.matmul(h_flat1,w_fc1,name='v_fc1')
                fc1=v_fc1+b_fc1
                fc1_mean, fc1_var = tf.nn.moments(
                    fc1,
                    axes=[0],
                )
                scale1 = tf.Variable(tf.ones([1]))
                shift1 = tf.Variable(tf.zeros([1]))
                epsilon1 = 0.001
                #ema = tf.train.ExponentialMovingAverage(decay=0.5)
                fc1_norm=tf.nn.batch_normalization(fc1, fc1_mean, fc1_var, shift1, scale1, epsilon1)
                h_fc1=tf.nn.relu(fc1_norm,name='h_fc1')
                #h_fc1 = tf.nn.relu(v_fc1+b_fc1, name='h_fc1')
                h_drop1=tf.nn.dropout(h_fc1,self.keep_prob[0],name='h_drop1')
            with tf.name_scope("fully_connected_layer2"):
                w_fc2=tf.Variable(tf.truncated_normal([1024,100],stddev=0.1),name='wfc2')
                b_fc2=tf.Variable(tf.constant(0.01,shape=[100]),name='b_fc2')
                v_fc2=tf.matmul(h_drop1,w_fc2,name='v_fc2')
                fc2 = v_fc2 + b_fc2
                fc2_mean, fc2_var = tf.nn.moments(
                    fc2,
                    axes=[0],
                )
                scale2 = tf.Variable(tf.ones([1]))
                shift2 = tf.Variable(tf.zeros([1]))
                epsilon2 = 0.001
                # ema = tf.train.ExponentialMovingAverage(decay=0.5)
                fc2_norm = tf.nn.batch_normalization(fc2, fc2_mean, fc2_var, shift2, scale2, epsilon2)
                h_fc2 = tf.nn.relu(fc2_norm, name='h_fc2')
                #h_fc2=tf.nn.relu6(v_fc2+b_fc2,name='h_fc2')
                h_drop2=tf.nn.dropout(h_fc2,self.keep_prob[1],name='h_drop2')
            with tf.name_scope("fully_connected_layer3"):
                w_fc3 = tf.Variable(tf.truncated_normal([100,self.size_y],stddev=0.1), name='wfc3')
                b_fc3 = tf.Variable(tf.constant(0.01, shape=[self.size_y]), name='b_fc3')
                v_fc3 = tf.matmul(h_drop2, w_fc3, name='v_fc3')
                h_fc3=tf.add(v_fc3,b_fc3,name='h_fc3')
                return h_fc3

    def cnn_def(self,input_x,scope='CNN',collect='CNN'):
        with tf.name_scope(scope):
            with tf.name_scope('Batch_norm'):
                fc_mean, fc_var = tf.nn.moments(
                    input_x,
                    axes=[0],
                )
                scale = tf.Variable(tf.ones([1]))
                shift = tf.Variable(tf.zeros([1]))
                epsilon = 0.001
                ema = tf.train.ExponentialMovingAverage(decay=0.5)

                def mean_var_with_update():
                    ema_apply_op = ema.apply([fc_mean, fc_var])
                    with tf.control_dependencies([ema_apply_op]):
                        return tf.identity(fc_mean), tf.identity(fc_var)

                mean, var = mean_var_with_update()
                input_x1 = tf.nn.batch_normalization(input_x, mean, var, shift, scale, epsilon)
            with tf.name_scope('layer1'):
                #[-1,1920,2560,3] -> [-1,1920,2560,3]
                lay_1_channel=3
                w_conv1=tf.Variable(tf.truncated_normal([3,3,3,lay_1_channel]),name='w_conv1')
                b_conv1=tf.Variable(tf.constant(0.1,shape=[lay_1_channel]),name='b_conv1')
                v_conv1=tf.nn.conv2d(input_x1,w_conv1,strides=[1,1,1,1],padding='SAME',name='v_conv1')
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
                h_fc3 = tf.nn.tanh(v_fc3 + b_fc3, name='h_fc3')
                h_drop3 = tf.nn.dropout(h_fc3, self.keep_prob[2], name='h_drop3')

            with tf.name_scope("fully_connected_layer4"):
                w_fc4 = tf.Variable(tf.truncated_normal([32,self.size_y],stddev=0.1), name='wfc4')
                b_fc4 = tf.Variable(tf.constant(0.05, shape=[self.size_y]), name='b_fc4')
                v_fc4 = tf.matmul(h_drop3, w_fc4, name='v_fc4')
                h_fc4=tf.add(v_fc4,b_fc4,name='h_fc4')
                #out=tf.nn.softmax(h_fc4)
                return h_fc4

    #def buidCNN(self):
    #    self.graph=tf.Graph()
    #    with self.graph.as_default():
    def generate_random_batch_from_aug_data(self,x,y):
        data_x=[]
        data_y=[]
        x_all=[]
        y_all=[]
        #y_all.extend(y)
        x_resized = data_augmentation(x)
        #x_all.extend(x_resized)
        #x_resized_scale, y_scale = central_scale_images(x_resized, y, [0.9, 0.75, 0.6])
        #_all.extend(x_resized_scale)
        #print(x_resized_scale.shape)
        #print(y_scale.shape)
        #y_all.extend(y_scale)
        x_flip, y_flip = flip_images(x_resized, y)
        #print(x_flip.shape)
        #print(y_flip.shape)
        x_all.extend(x_flip)
        y_all.extend(y_flip)
        #x_transpose, y_transpose = translate_images(x_resized, y)
        #print(x_transpose.shape)
        #print(y_transpose.shape)
        #x_all.extend(x_transpose)
        #y_all.extend(y_transpose)
        x_rotate,y_rotate=rotate_images(x_resized,y)
        x_all.extend(x_rotate)
        y_all.extend(y_rotate)
        x_gaussian,y_gaussian=add_gaussian_noise(x_resized,y)
        x_all.extend(x_gaussian)
        y_all.extend(y_gaussian)
        shuf = np.random.choice([i for i in range(len(x_all))],30,replace=False)
        for i in shuf:
            data_x.append(x_all[i])
            data_y.extend([y_all[i]])
        data_x.extend(x_resized)
        data_y.extend(y)
        return np.asarray(data_x),np.asarray(data_y)
    def train_para(self):
        with self.graph.as_default():
            with tf.name_scope("loss"):
                self.label=self.one_hot_label(self.input_y)
                #print(self.input_y.get_shape())
                #print(self.input_x.get_shape())
                #print(self.eval_logit.get_shape())
                #self.cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.eval_logit,name='cross_entropy')
                self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.input_y,self.size_y),logits=self.eval_logit)
                #self.cross_entropy=tf.one_hot(self.input_y,self.size_y)*tf.log(tf.nn.softmax(self.eval_logit))
                #self.cross_entropy_mean=-tf.reduce_mean(self.cross_entropy)

                self.cross_entropy_mean=tf.reduce_mean(self.cross_entropy,name='cross_entropy_mean')
                vars = tf.trainable_variables()
                self.lossL2=tf.add_n([tf.nn.l2_loss(v) for v in vars])
                #regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                #self.regularization=tf.reduce_sum(regularization_losses)
                #self.loss=self.cross_entropy_mean+0.001*tf.reduce_sum(regularization_losses)
                self.loss = self.cross_entropy_mean + 0.0002*self.lossL2

            with tf.name_scope("train"):
                self._lr=tf.Variable(0.0,trainable=False)
                self._new_lr=tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
                self._lr_update=tf.assign(self._lr,self._new_lr)
                #initial_learning_rate=0.5
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                #self.learning_rate = tf.train.exponential_decay(initial_learning_rate,
                #                                          global_step=self.global_step,
                #                                           decay_steps=10, decay_rate=0.9)

                #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.optimizer=tf.train.GradientDescentOptimizer(self._lr)
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.cross_entropy_mean, tvars), 5000)
                self.train_step=self.optimizer.minimize(self.loss,global_step=self.global_step)
                #self.train_step = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
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
            lr_init = 0.5
            lr_end = 1e-3
            i = 0
            run_step=0
            train_times = 20
            rank=0
            acc=0

            #print (self.train_len)
            #train, train_label, test, test_label = self.split_data()
            x_test,y_test=self.get_validate_data_from_file()
            x_test_resized=data_augmentation_test(x_test)
            #print(x_test_resized.shape)
            name,predict_data=self.get_test_data()
            predict_data_resized=data_augmentation_test(predict_data)
            while run_step<500 and acc<=0.95:
                #print (run_step)
                if i <  20:
                    cur_lr = lr_init
                else:
                    learning_rate=lr_init*(0.9**(run_step/20))
                    cur_lr = learning_rate
                sess.run(self._lr_update,feed_dict={self._new_lr: cur_lr})
                #x=train
                #y=train_label
                #if run_step<20:
                x,y=self.get_batch_data_from_file(rank)
                #x, y = self.get_even_distrib_batch_data(rank)
                x_,y_=self.generate_random_batch_from_aug_data(x,y)

                #print (x_distorted.shape)
                #else:
                #x,y=self.get_even_distrib_batch_data(rank)
                #print(x.shape)
                #print(y)
                rank=rank+1
                #print (x.shape)
                #print (y)
                #print (test)
                #print(y)
                #x_test=test[0]
                #y_test=test[1]
                sess.run(self.train_step, feed_dict={self.input_x: x_, self.input_y: y_})
                loss = sess.run(self.loss, feed_dict={self.input_x: x_, self.input_y: y_})
                train_acc = sess.run(self.accuracy, feed_dict={self.input_x: x_, self.input_y: y_})
                #self.batch=len(x_test)
                #print (len(x_test))
                test_loss = sess.run(self.loss, feed_dict={self.input_x: x_test_resized, self.input_y: y_test})
                test_acc = sess.run(self.accuracy, feed_dict={self.input_x: x_test_resized, self.input_y: y_test})
                acc=test_acc
                #self.batch=len(x)
                #reg=sess.run(self.lossL2,feed_dict={self.input_x: x_resized, self.input_y: y})
                #print (reg)
                if run_step % 5 == 0:
                    print("run_step: %g, train_loss: %g, test_loss: %g, train_acc: %g, test_acc: %g" % (
                    run_step, loss, test_loss, train_acc, test_acc))
                run_step = run_step + 1;
                i=i+1;
            predict_label=sess.run(self.predict,feed_dict={self.input_x: predict_data_resized})
            label=self.turn_num_to_label(predict_label)
            self.write_submit_file(name,label)
            #print (self.turn_num_to_label(predict_label))
        return 0
    def write_submit_file(self,jpg,label):
        mat=np.array([jpg,label])
        df=pd.DataFrame(mat.transpose(),columns=['jpg','prediction']);
        df_sorted=df.sort_values('jpg')
        #filename="my_submit.csv"
        df_sorted.to_csv("my_submit.csv",index=False,header=False)



def main_proc():
    gd=GD_CNN();
    gd.train_para()
    gd.train()

if __name__=="__main__":
    main_proc()



