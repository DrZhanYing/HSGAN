# -*- coding: utf-8 -*-
"""

"""

from __future__ import print_function
import numpy as np
#np.random.seed(1337)#it will generate the same random number array
from keras.utils  import np_utils
import random
from scipy import io




def getOneClass(data,label,classNO):
    
    if data.shape[1]==1:
        data=data.reshape([data.shape[0],data.shape[2]])
    dataBandNumber=data.shape[1]
    
    nbClassNumber=0
    for i in range(data.shape[0]):
        if(label[i]==classNO):
            nbClassNumber+=1
    #print(nbClassNumber)
    
    dataOneClass=np.zeros([nbClassNumber,dataBandNumber])#
    
    nbClassNumber=0
    for i in range(data.shape[0]):
        if(label[i]==classNO):
            dataOneClass[nbClassNumber,:]=data[i,:]
            nbClassNumber+=1

    return dataOneClass







def getDataindiafrom_mat():
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
    
    
    indexofclass=np.array((1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))#class number
    dictofclass={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0}
    
    
    
    
    index=0
    for i in range(145):
        for j in range(145):        
            if (l[i,j]!=0):           
                dataNormal[index,0,:]=d[i,j,:]
                dataOringin[index,0,:]=dOri[i,j,:]
                label[index]=l[i,j]-1

                dictofclass[label[index]] += 1            
                index += 1
    
    
    
    return dataNormal,dataOringin,label




def main():
    datan,datao,label=getDataindiafrom_mat()
    a=getOneClass(datan,label,1)


def india2():
    datan,datao,label=getDataindiafrom_mat()
    datanAll=datan.reshape((datan.shape[0],200))
    return datanAll
    
    
    