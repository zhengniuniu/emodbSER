# -*- coding:utf-8 -*-
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Flatten,Dropout,Activation,Flatten
from keras.models import Model,Sequential
from keras.datasets import mnist
import theano
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from keras.utils import np_utils
from PIL import Image
from random import shuffle
import os
import cPickle

np.random.seed(1337) # for reproducibililty

"""def load_data():
    data=np.empty((535,1,513,77),dtype="float32")
    label=np.empty(535,dtype=int)
    
    imgs=os.listdir('/home/krumo/Documents/EMO-DB/spec/')
    num=len(imgs)
    for i in range(num):
        img=Image.open('/home/krumo/Documents/EMO-DB/spec/'+imgs[i])
        arr=np.asarray(img,dtype="float32")
        data[i,:,:,:]=arr
        if imgs[i][5]=='W':
            label[i]=1
        elif imgs[i][5]=='L':
            label[i]=2
        elif imgs[i][5]=='E':
            label[i]=3
        elif imgs[i][5]=='A':
            label[i]=4
        elif imgs[i][5]=='F':
            label[i]=5   
        elif imgs[i][5]=='T':
            label[i]=6
        elif imgs[i][5]=='N':
            label[i]=0            
        else:
            label[i]=0
            print "error:"+imgs[i]
    return data,label"""
def generate_label(X_Label,data,label_W,label_L,label_E,label_A,label_F,label_T,label_N):
        
    W=label_W
    shuffle(W)
    L=label_L
    shuffle(L)
    E=label_E
    shuffle(E)
    A=label_A
    shuffle(A)
    F=label_F
    shuffle(F)
    T=label_T
    shuffle(T)
    N=label_N
    shuffle(N)
    for i in range(len(label_A)):
        if label_A[i] <464:
            X_Label[label_A[i],:]=data[A[i],:]
    for i in range(len(label_E)):
        if label_E[i] <464:
            X_Label[label_E[i],:]=data[E[i],:]    
    for i in range(len(label_F)):
        if label_F[i] <464:
            X_Label[label_F[i],:]=data[F[i],:]
     for i in range(len(label_L)):
        if label_L[i] <464:
            X_Label[label_L[i],:]=data[L[i],:]
     for i in range(len(label_N)):
        if label_N[i] <464:
            X_Label[label_N[i],:]=data[N[i],:]
     for i in range(len(label_T)):
        if label_T[i] <464:
            X_Label[label_T[i],:]=data[T[i],:]
     for i in range(len(label_W)):
        if label_W[i] <464:
            X_Label[label_W[i],:]=data[W[i],:]
     return X_Label
            
            
    
def load_data():
    f=open('../emodb.pkl','rb')
    data,label,label_W,label_L,label_E,label_A,label_F,label_T,label_N=cPickle.load(f)
    
    X_train=np.empty((464,img_cols*img_rows))
    X_train1=np.empty((464,img_cols*img_rows))
    X_train2=np.empty((464,img_cols*img_rows))
    X_train3=np.empty((464,img_cols*img_rows))
    X_train4=np.empty((464,img_cols*img_rows))
    Y_train=np.empty(464,dtype=int)
    
    X_valid=np.empty((535,img_cols*img_rows))
    Y_valid=np.empty(535,dtype=int)
    
    X_test=np.empty((71,img_cols*img_rows))
    Y_test=np.empty(71,dtype=int)
    
    for i in range(535):
        if i<464:
            X_train[i,:]=data[i,:]
            Y_train[i]=label[i]
        else:
            X_test[i-464,:]=data[i,:]
            Y_test[i-464]=label[i]
        X_valid[i,:]=data[i,:]
        Y_valid[i]=label[i]
    X_train1=generate_label(X_train1,data,label_W,label_L,label_E,label_A,label_F,label_T,label_N)
    X_train2=generate_label(X_train2,data,label_W,label_L,label_E,label_A,label_F,label_T,label_N)
    X_train3=generate_label(X_train3,data,label_W,label_L,label_E,label_A,label_F,label_T,label_N)
    X_train4=generate_label(X_train4,data,label_W,label_L,label_E,label_A,label_F,label_T,label_N)
    return (X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
                                    
"""(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)"""


if __name__=='__main__':
    batch_size = 10
    nb_classes = 7
    nb_epoch = 1

    # input image dimensions
    img_rows, img_cols = 516, 76
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    """(X_train, Y_train, X_valid, Y_valid, X_test, Y_test) = split_data('G:\data\olivettifaces.pkl')
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)"""
    x_train,y_train,x_valid,y_valid,x_test,y_test=load_data()

    x_train = x_train.astype('float32') / 255.0
    X_train = np.reshape(x_train, (len(x_train), 1, img_rows,img_cols))

    x_test = x_test.astype('float32') / 255.0
    X_test = np.reshape(x_test, (len(x_test), 1, img_rows,img_cols))

    x_valid = x_valid.astype('float32') / 255.0
    X_valid = np.reshape(x_valid, (len(x_valid), 1, img_rows,img_cols))

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    # convert label to binary class matrix

    model = Sequential()
    # 32 convolution filters , the size of convolution kernel is 3 * 3
    # border_mode can be 'valid' or 'full'
    #‘valid’only apply filter to complete patches of the image. 
    # 'full'  zero-pads image to multiple of filter shape to generate output of shape: image_shape + filter_shape - 1
    # when used as the first layer, you should specify the shape of inputs 
    # the first number means the channel of an input image, 1 stands for grayscale imgs, 3 for RGB imgs
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='same',
                            input_shape=(1, img_rows, img_cols)))
    # use rectifier linear units : max(0.0, x)
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # second convolution layer with 32 filters of size 3*3
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    # max pooling layer, pool size is 2 * 2
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # drop out of max-pooling layer , drop out rate is 0.25 
    model.add(Dropout(0.25))
    # flatten inputs from 2d to 1d
    model.add(Flatten())
    # add fully connected layer with 128 hidden units
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # output layer with softmax 
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # use cross-entropy cost and adadelta to optimize params
    

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_valid, Y_valid))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
