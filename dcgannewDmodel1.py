# -*- coding: utf-8 -*-



from keras.models import Sequential
from keras.layers import Dense, Input,Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
#from PIL import Image
import argparse
import math
from indiafunction import india2
import matplotlib.pyplot as plt


from keras.models import Model

from keras import backend as K
K.set_image_dim_ordering('th')

#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')




def generator_model():
    #原始G
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*1*50))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 1, 50), input_shape=(128*1*50,)))
    model.add(UpSampling2D(size=(1, 2)))
    model.add(Convolution2D(64, 1, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(1, 2)))
    model.add(Convolution2D(1, 1, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model



def generator_modelNew1():

    g_input = Input(shape=[100])
   
    H = Dense(1024)(g_input)
    H = Activation('tanh')(H)

    H = Dense(612*1*50)(g_input)
    H = BatchNormalization()(H)
    H = Activation('tanh')(H)
    H = Reshape( [612, 1,50] )(H)
    H = UpSampling2D(size=(1, 2))(H)
    H = Convolution2D(128, 1, 2, border_mode='same')(H)
    H = BatchNormalization()(H)
    H = Activation('tanh')(H)
    H = UpSampling2D(size=(1, 2))(H)
    H = Convolution2D(32, 1, 3, border_mode='same')(H)
    H = BatchNormalization()(H)
    H = Activation('tanh')(H)
    H = Convolution2D(1, 1, 1, border_mode='same')(H)
    g_V = Activation('tanh')(H)
    generator = Model(g_input,g_V)
    #generator.compile(loss='binary_crossentropy', optimizer='SGD')
    #generator.summary()
    return generator





def discriminator_modelORI():
    #原始的D
    model = Sequential()
    model.add(Convolution2D(
                        64, 1, 5,
                        border_mode='same',
                        input_shape=(1, 1, 200)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Convolution2D(128, 1, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model



def discriminator_model():

    model = Sequential()
    model.add(Convolution2D(
                        32, 1, 3,
                        border_mode='same',
                        input_shape=(1, 1, 200)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,1,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(0.25)) 
    model.add(Convolution2D(32,1,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,1,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model



def discriminator_model16():

    model = Sequential()
    model.add(Convolution2D(
                        32, 1, 3,
                        border_mode='same',
                        input_shape=(1, 1, 200)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,1,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(0.25)) 
    model.add(Convolution2D(32,1,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(32,1,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.summary()
    return model



def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model




def combine_images(generated_images):
    num = generated_images.shape[0]#128
    width = 200#11
    height = 128#12
    shape = generated_images.shape[2:]#(1,200)
    
    #image=()
    image = np.zeros((128, 200), dtype=generated_images.dtype)
    
    #index=0--127,img=(1,1,200)
    for index, img in enumerate(generated_images):
        image[index, :] = img[0, 0, :]
    return image



def combine_images200(generated_images):
    num = generated_images.shape[0]#128
    width = 200#11
    height = 128#12
    shape = generated_images.shape[2:]#(1,200)
    
    #image=()
    image = np.zeros((128, 200), dtype=generated_images.dtype)
    
    #index=0--127,img=(1,1,200)
    for index, img in enumerate(generated_images):
        image[index, :] = img[0, 0, :]
    
    
    return image

def train(BATCH_SIZE):
    


    dataAll=india2()    
    data=dataAll.reshape((dataAll.shape[0],1,1,200))

    
    
    X_train=data
    

    X_train = (X_train.astype(np.float32) - 0.5)/0.5
    
    
    discriminator = discriminator_model()
    generator = generator_model()
    
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    
    
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    
    discriminator.trainable = True
    
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    
    noise = np.zeros((BATCH_SIZE, 100))#(128,100)
    
    
    #fd = file("dloss.txt", "wb")
    #fg = file("gloss.txt", "wb")
    
    for epoch in range(500):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))

        
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            

            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
 
            generated_images = generator.predict(noise, verbose=0)
            
  
            X = np.concatenate((image_batch, generated_images))
            
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE


            d_loss = discriminator.train_on_batch(X, y)
            
            print("batch %d d_loss : %f" % (index, d_loss))
            
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)#
                
            discriminator.trainable = False
            

            
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            
            
            if index % 10 == 0 and epoch < 20:
                image = combine_images(generated_images)#
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+"_g_"+str(g_loss)+"_d_"+str(d_loss)+"_"+".png")
                np.savez(str(epoch)+"_"+str(index)+".npz",generated_images)
                
            if index % 100 == 0 and epoch >= 20:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+"_g_"+str(g_loss)+"_d_"+str(d_loss)+"_"+".png")
                np.savez(str(epoch)+"_"+str(index)+".npz",generated_images)
            

            
            
            
            discriminator.trainable = True
            
            print("batch %d g_loss : %f" % (index, g_loss))
            
            if index % 100 == 0 and epoch % 10 == 0:
                generator.save_weights(str(epoch)+"_"+str(g_loss)+"_"+'generatorALL', True)#
                discriminator.save_weights(str(epoch)+"_"+str(d_loss)+"_"+'discriminatorALL', True)#save the weights not the model structure
    
    
    #fd.close()
    #fg.close()
    
def train10(BATCH_SIZE):

    
    aaa=np.load("indiareadorigin10class_200onevector_normalize.npz")

    data=aaa["arr_0"]


    
    data10=data
    
    data=data10.reshape((data10.shape[0],1,1,200))
    
   
    
    X_train=data

    X_train = (X_train.astype(np.float32) - 0.5)/0.5
    
    
    discriminator = discriminator_model()
    generator = generator_model()
    
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    
    
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(3000):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
               
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            
            generated_images = generator.predict(noise, verbose=0)
            
            #
            if index % 100 == 0 and epoch < 500:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
                np.savez(str(epoch)+"_"+str(index)+".npz",generated_images)
            
            if index % 100 == 0 and epoch > 500 and epoch % 10 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch)+"_"+str(index)+".png")
                np.savez(str(epoch)+"_"+str(index)+".npz",generated_images)             
                
                
                
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            

            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))

            
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)#
                
            discriminator.trainable = False
            
            #G
            
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            
            discriminator.trainable = True
            
            print("batch %d g_loss : %f" % (index, g_loss))
            
            if index % 100 == 0:
                generator.save_weights('generator10w', True)#
                generator.save('generator10')
                discriminator.save_weights('discriminator10w', True)
                discriminator.save('discriminator10')

def generate_old(BATCH_SIZE, nice=False):
    generator = generator_model()
    
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE*20, 100))
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")




def generate1(BATCH_SIZE, nice=True):

    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    
    generator.load_weights('generator1')
    
    print("A***************************")
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator1')
        noise = np.zeros((BATCH_SIZE*20, 100))#2560 100
        
    
        
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        
        generated_images = generator.predict(noise, verbose=1)
        
        d_pret = discriminator.predict(generated_images, verbose=1)
        
        index = np.arange(0, BATCH_SIZE*20)
        
        index.resize((BATCH_SIZE*20, 1))#
        pre_with_index = list(np.append(d_pret, index, axis=1))#
        pre_with_index.sort(key=lambda x: x[0], reverse=True)#
        

        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")
    return image

def generateALL(BATCH_SIZE, nice=True):

    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    
    generator.load_weights('generatorALL')
    
    print("A***************************")
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator1')
        noise = np.zeros((BATCH_SIZE*20, 100))#2560 100
        
    
        
        for i in range(BATCH_SIZE*20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        
        generated_images = generator.predict(noise, verbose=1)
        
        d_pret = discriminator.predict(generated_images, verbose=1)#(2560, 1)
        
        index = np.arange(0, BATCH_SIZE*20)#array([   0,    1,    2, ..., 2557, 2558, 2559])
        
        index.resize((BATCH_SIZE*20, 1))#(2560,1)
        
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        
        #nice_images=((128, 1, 1, 200))
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")
    return image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=True)
    args = parser.parse_args()
    return args




def plotimageold():    
    plt.close('all')    



def plotimage(image,name):  


    plt.close('all')
    

    f, axarr = plt.subplots(2, 2)
    

    for i in range(2):
        for j in range(2):
            a=np.random.randint(0,image.shape[0])
            print(a)
            axarr[i,j].plot(image[a,:])            
    
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
    
    plt.savefig(name)



train(128)#

#train10(128)
        
            
            
            
            
            
            
            
            
            
