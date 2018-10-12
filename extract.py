#!/usr/bin/env python3
import  zipfile  as zf
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
import os
import cv2
os.chdir("/home/jiangwei/home/jiangwei/home/jiangwei/kaggle")

class  GetPngData(object):

    def __init__(self):
        self.path="/home/jiangwei/home/jiangwei/home/jiangwei/kaggle"
        self.train_name="guangdong_round1_train1_20180903.zip"
        self.train_name2="guangdong_round1_train2_20180916.zip"
        self.test_name="guangdong_round1_test_a_20180910.zip"
        self.name_2_id={}
        self.train_path=join(self.path,self.train_name)
        self.train_path2=join(self.path,self.train_name2)
        self.test_path=join(self.path,self.test_name)
        self.train_data_path=join(self.path,"guangdong_round1_train1_20180903")
        self.train_data_path2=join(self.path,"guangdong_round1_train2_20180916")
        self.test_data_path=join(self.path,"guangdong_round1_test_a_20180910")
    def extract_train(self):
        file=zf.ZipFile(self.train_path,'r')
        for name in file.namelist():
            #print(1)
            path=Path(file.extract(name))
            #print(path)
            path.rename(name.encode('cp437').decode('gbk'))
        dir=file.namelist()[0]
        file.close()
        self.train_data_path=join(self.path,dir)

            #e.encode('cp437').decode('gbk'))
            #print(path)
    def extract_train2(self):
        file=zf.ZipFile(self.train_path2,'r')
        for name in file.namelist():
            #print(1)
            path=Path(file.extract(name))
            #print(path)
            path.rename(name.encode('cp437').decode('gbk'))
        dir=file.namelist()[0]
        file.close()
        self.train_data_path2=join(self.path,dir)
        #print (self.train_data_path2)
    def extract_test(self):
        #self.test_name=[]
        file=zf.ZipFile(self.test_path,'r')
        for name in file.namelist():
            #self.test_name.append(name)
            path=Path(file.extract(name))
            path.rename(name.encode('cp437').decode('gbk'))
        dir = file.namelist()[0]
        file.close()
        self.test_data_path = join(self.path, dir)
        print(self.test_data_path)


    def get_other_category_data(self,category,path):
        map_dict={}
        label_dict={}
        for x in os.listdir(path):
            if  x.endswith(".jpg"):
                map_dict[x]=join(path,x)
                label_dict[x]=category
        return map_dict,label_dict
    def get_train_data2(self):

        path_list = os.listdir(self.train_data_path2)

        #category=["瑕疵样本"]
        map_label_dict = {}
        map_name_dict = {}
        for i in range(0,len(path_list)):
            path_name=path_list[i]
            path=join(self.train_data_path2,path_name);
            if len(os.listdir(path))==0:
                print(' '.join([path,"is empty"]))
            else:
                list_cat=os.listdir(path)
                if os.path.isdir(join(path,list_cat[0])):
                    for j in range(len(list_cat)):
                        pn_list=os.listdir(join(path,list_cat[j]));
                        for p in range(len(pn_list)):
                            new_path=join(path,list_cat[j],pn_list[p])
                            if os.path.isdir(join(path,list_cat[j],pn_list[p])):
                                other_data,other_label=self.get_other_category_data(list_cat[j],new_path)
                                map_name_dict.update(other_data)
                                map_label_dict.update(other_label)
                            else:
                                if pn_list[p].endswith(".jpg"):
                                    label = list_cat[j]
                                    map_name_dict[pn_list[p]] = join(path,list_cat[j],pn_list[p])
                                    map_label_dict[pn_list[p]] = label
                else:
                    for l in range(len(list_cat)):
                        label= path_name
                        map_name_dict[list_cat[l]] = join(path, list_cat[l])
                        map_label_dict[list_cat[l]] = label
        return [map_name_dict, map_label_dict]
    def get_train_data(self):
        pn_list=os.listdir(self.train_data_path)
        map_label_dict={}
        map_name_dict={}
        for i in range(len(pn_list)):
            label=pn_list[i][0:2]
            map_name_dict[i]=pn_list[i]
            map_label_dict[i]=label
        return [map_name_dict,map_label_dict]
    def get_test_data(self):
        pn_list=os.listdir(self.test_data_path)
        map_name_dict={}
        for i in range(len(pn_list)):
            map_name_dict[i]=pn_list[i]
        return map_name_dict
    def read_train_img2(self,train_map,label_map):
        train_data=[]
        label_data=[]
        for k in train_map.keys():
            #pic_name=join(self.train_data_path2,train_map[i])
            pic_name=train_map[k]
            #print (pic_name)
            train_data.append(cv2.imread(pic_name))
            label_data.append(label_map[k])
        label_num=self.turn_label_to_num(label_data)
        #df_train=pd.DataFrame(train_data)
        return [train_data,label_num]

    def read_train_img(self,train_map,label_map):
        train_data=[]
        label_data=[]
        for i in range(len(train_map)):
            pic_name=join(self.train_data_path,train_map[i])
            #print (pic_name)
            train_data.append(cv2.imread(pic_name))
        for i in range(len(label_map)):
            label_data.append(label_map[i])
        label_num=self.turn_label_to_num(label_data)
        #df_train=pd.DataFrame(train_data)
        return [train_data,label_num]

    def read_test_img(self,test_map):
        test_data=[]
        test_name=[]
        for i in range(len(test_map)):
            test_name.append(test_map[i])
            pic_name=join(self.test_data_path,test_map[i])
            test_data.append(cv2.imread(pic_name))
        return test_name,test_data

    def turn_label_to_num(self,label):

        label_dict={'无瑕疵样本':0, '不导电':1, '凸粉':2, '擦花':3,'桔皮' :4,'横条压凹':5,'涂层开裂':6, '漏底':7,'碰伤':8,'脏点':9, '起坑':10,'其他':11}
        #df_label=pd.DataFrame(label)
        #label_num=df_label[0].map(label_dict)
        label_num=[label_dict[k] for k in label]
        return label_num

    def main_train(self):
        #self.extract_train2()
        train_map,label_map=self.get_train_data2()
        train,label=self.read_train_img2(train_map,label_map)
        return [train,label]
    def main_test(self):
        #self.extract_test()
        test_map=self.get_test_data()
        name,data=self.read_test_img(test_map)
        return name,data


if __name__=="__main__":
    Png_Data=GetPngData()
    x,y=Png_Data.get_train_data2()
    name,data=Png_Data.main_test()
    print(len(x))
    #print(len(data))
    #print(name)
    #x,y=Png_Data.main()
    #print (os.listdir(os.getcwd()))
