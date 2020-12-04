#!/usr/bin/env python2
# -*- coding: utf-8 -*-



from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils  import np_utils

from sklearn.cross_validation import train_test_split

from dcgannewDmodel1 import discriminator_model 
from scipy import io

from keras import backend as K
K.set_image_dim_ordering('th')


d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
bili=5
eps=5000
savefile='loadDmodel16class3w5samples_00005_5000e_190.h5'

def getDataindiafrom_mat(random=False):
    datafilepath="Indian_pines_corrected.mat"
    labelfilepath="Indian_pines_gt.mat"
    
    a=io.loadmat(datafilepath)
    aa=io.loadmat(labelfilepath)
    d=a['indian_pines_corrected']
    dOri=a['indian_pines_corrected']

    l=aa['indian_pines_gt']
    d=np.float32(d)
    
    
    
    d /= d.max()
    

    
    dataNormal=np.empty((10249,1,200),dtype="float32")
    dataOringin=np.empty((10249,1,200),dtype="float32")
    label=np.empty((10249),dtype="int32")
    
    
    indexofclass=np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
    dictofclass={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
   
    
    
    index=0
    for i in range(145):#find the no.14 class
        for j in range(145):        
            if (l[i,j]!=0):           
                dataNormal[index,0,:]=d[i,j,:]#
                dataOringin[index,0,:]=dOri[i,j,:]#
                label[index]=l[i,j]-1#
                
                dictofclass[label[index]] += 1 #      
                index += 1
    
    if(random):    
        np.random.seed(1111)
        np.random.shuffle(dataNormal)
        np.random.seed(1111)
        np.random.shuffle(dataOringin)
        np.random.seed(1111)
        np.random.shuffle(label)
    
    
    
    return dataNormal,dataOringin,label

def getOneClass2(data,label,classNO):
    
    if data.shape[1]==1:
        data=data.reshape([data.shape[0],data.shape[2]])
    dataBandNumber=data.shape[1]
    
    nbClassNumber=0
    for i in range(data.shape[0]):
        if(label[i]==classNO):
            nbClassNumber+=1
    #print(nbClassNumber)
    
    dataOneClass=np.zeros([nbClassNumber,dataBandNumber])#
    labelone=np.zeros([nbClassNumber])#
    labelone=labelone+classNO
    
    nbClassNumber=0
    for i in range(data.shape[0]):
        if(label[i]==classNO):
            dataOneClass[nbClassNumber,:]=data[i,:]
            nbClassNumber+=1

    return dataOneClass, labelone

def train_test_split_percentofclass(data,label,train_size,rs=0):
    nbclass=16
    nbbands=200
    
    Datatest=np.zeros((0,200))
    Datatrain=np.zeros((0,200))
    Labeltest=np.zeros((0))
    Labeltrain=np.zeros((0))
    
    nbTrainTestAll=np.zeros((16,3),dtype="int32")
    
    
    for i in range(nbclass):
        dataone,labelone=getOneClass2(data,label,i)
        
        #X(n,200) y(n)
        Dtest,Dtrain,Ltest,Ltrain = train_test_split(dataone,labelone,test_size = train_size,random_state = rs)
        
        Datatest=np.concatenate((Datatest,Dtest), axis=0)
        Datatrain=np.concatenate((Datatrain,Dtrain), axis=0)
        Labeltest=np.concatenate((Labeltest,Ltest), axis=0)
        Labeltrain=np.concatenate((Labeltrain,Ltrain), axis=0)
        
        nbTrainTestAll[i,0] = Ltrain.shape[0]
        nbTrainTestAll[i,1] = Ltest.shape[0]
        nbTrainTestAll[i,2] = Ltrain.shape[0] + Ltest.shape[0]
    
    return Datatrain,Datatest,Labeltrain,Labeltest,nbTrainTestAll
        
        
        
        

def train_test_split_nbSamplesofclass(data,label,trainnb=5,rs=0):
    nbclass=16
    nbbands=200
    
    Datatest=np.zeros((0,200))
    Datatrain=np.zeros((0,200))
    Labeltest=np.zeros((0))
    Labeltrain=np.zeros((0))
    
    nbTrainTestAll=np.zeros((16,3),dtype="int32")
    
    
    for i in range(nbclass):
        dataone,labelone=getOneClass2(data,label,i)
        
        #X(n,200) y(n)
        Dtest,Dtrain,Ltest,Ltrain = train_test_split(dataone,labelone,test_size = trainnb,random_state = rs)
        
        Datatest=np.concatenate((Datatest,Dtest), axis=0)
        Datatrain=np.concatenate((Datatrain,Dtrain), axis=0)
        Labeltest=np.concatenate((Labeltest,Ltest), axis=0)
        Labeltrain=np.concatenate((Labeltrain,Ltrain), axis=0)
        
        nbTrainTestAll[i,0] = Ltrain.shape[0]
        nbTrainTestAll[i,1] = Ltest.shape[0]
        nbTrainTestAll[i,2] = Ltrain.shape[0] + Ltest.shape[0]
        
    #random
    
    return Datatrain,Datatest,Labeltrain,Labeltest,nbTrainTestAll







D = discriminator_model()



D.load_weights('190_0.245788_discriminatorALL')

D.trainable = True



D.pop()#
D.pop()#
D.pop()#
D.pop()#
D.pop()#
D.pop()#



dataN,dataO,label=getDataindiafrom_mat(random=True)
dataN= (dataN.astype(np.float32) - 0.5)/0.5 # normorlize to 0
Datatrain,Datatest,Labeltrain,Labeltest,nbTrainTestAll=train_test_split_percentofclass(dataN,label,bili,rs=0)
#Datatrain,Datatest,Labeltrain,Labeltest,nbTrainTestAll=train_test_split_nbSamplesofclass(dataN,label,5,rs=0)

#随机
np.random.seed(1111)#
np.random.shuffle(Datatrain)
np.random.seed(1111)#
np.random.shuffle(Labeltrain)#
np.random.seed(1111)
np.random.shuffle(Datatest)
np.random.seed(1111)
np.random.shuffle(Labeltest)





Datatrain=Datatrain.reshape((Datatrain.shape[0],1,1,200))
Datatest=Datatest.reshape((Datatest.shape[0],1,1,200))
#得到样本经过model的特征
bottleneck_features_train = D.predict(Datatrain)
bottleneck_features_validation=D.predict(Datatest)
Labeltrain16=np_utils.to_categorical(Labeltrain, 16)
Labeltest16=np_utils.to_categorical(Labeltest, 16)



model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16)) 
model.add(Activation('softmax'))




model.compile(loss='categorical_crossentropy', optimizer=d_optim, metrics=['accuracy'])


model.fit(bottleneck_features_train, Labeltrain16,
          epochs=eps,
          batch_size=32,
          validation_data=(bottleneck_features_validation, Labeltest16))

model.save_weights(savefile)
#




