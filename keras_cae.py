# -*- coding:utf-8 -*-
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Flatten
from keras.models import Model
from keras.datasets import mnist
import theano
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from keras.utils import np_utils
from random import shuffle
from PIL import Image
import cPickle
import os


"""def load_data():
    data=np.empty((535,1,516,76),dtype="float32")
    label=np.empty((535,),dtype="uint8")
    
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
img_rows=512
img_cols=128
batches=10
auto_epoch=1
classify_epoch=1
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
            
            
    
def load_data(file):
    f=open(file,'rb')
    data,label,label_W,label_L,label_E,label_A,label_F,label_T,label_N=cPickle.load(f)
    f.close()
    
    X_train=np.empty((464,img_cols*img_rows))
    X_train1=np.empty((464,img_cols*img_rows))
    X_train2=np.empty((464,img_cols*img_rows))
    X_train3=np.empty((464,img_cols*img_rows))
    X_train4=np.empty((464,img_cols*img_rows))
    Y_train=np.empty(464,dtype=int)
    
    X_valid=np.empty((len(data),img_cols*img_rows))
    Y_valid=np.empty(len(data),dtype=int)
    
    X_test=np.empty((71,img_cols*img_rows))
    Y_test=np.empty(71,dtype=int)
    
    for i in range(len(data)):
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
    #error check
    for i in range(len(X_train1)):
        m=0;
        for j in range(len(X_train1[i])):
            if X_train1[i][j]<>0:
                m=m+1
                break
        if m==0:
            print 'error: train1'
    for i in range(len(X_train2)):
        m=0;
        for j in range(len(X_train2[i])):
            if X_train2[i][j]<>0:
                m=m+1
                break
        if m==0:
            print 'error: train2'
    for i in range(len(X_train3)):
        m=0;
        for j in range(len(X_train3[i])):
            if X_train3[i][j]<>0:
                m=m+1
                break
        if m==0:
            print 'error: train3'
    for i in range(len(X_train4)):
        m=0;
        for j in range(len(X_train4[i])):
            if X_train1[i][j]<>0:
                m=m+1
                break
        if m==0:
            print 'error: train4'

    return (X_train,Y_train,X_valid,Y_valid,X_test,Y_test,X_train1,X_train2,X_train3,X_train4)

def load_pretrain(file):
    f=open(file,'rb')
    data,label,label_W,label_L,label_E,label_A,label_F,label_T,label_N=cPickle.load(f)
    f.close()
    
    X_train=np.empty((len(data),img_cols*img_rows))
    Y_train=np.empty(len(data),dtype=int)
    
    for i in range(len(data)):
        X_train[i,:]=data[i,:]
        Y_train[i]=label[i]
    return X_train,Y_train
    
"""(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)"""

x_train,y_train,x_valid,y_valid,x_test,y_test,x_train1,x_train2,x_train3,x_train4=load_data('./emodb.pkl')
enterface_train,enterface_label=load_pretrain('./enterface.pkl')
savee_train,savee_label=load_pretrain('./SAVEE.pkl')


x_train = x_train.astype('float32') / 255.0
x_train = np.reshape(x_train, (len(x_train), 1, img_rows,img_cols))

x_train1 = x_train1.astype('float32') / 255.0
x_train1 = np.reshape(x_train1, (len(x_train1), 1, img_rows,img_cols))

x_train2 = x_train2.astype('float32') / 255.0
x_train2 = np.reshape(x_train2, (len(x_train2), 1, img_rows,img_cols))

x_train3 = x_train3.astype('float32') / 255.0
x_train3 = np.reshape(x_train3, (len(x_train3), 1, img_rows,img_cols))

x_train4 = x_train4.astype('float32') / 255.0
x_train4 = np.reshape(x_train4, (len(x_train4), 1, img_rows,img_cols))

enterface_train = enterface_train.astype('float32') / 255.0
enterface_train = np.reshape(enterface_train, (len(enterface_train), 1, img_rows,img_cols))

savee_train = savee_train.astype('float32') / 255.0
savee_train = np.reshape(savee_train, (len(savee_train), 1, img_rows,img_cols))

x_test = x_test.astype('float32') / 255.0
x_test = np.reshape(x_test, (len(x_test), 1, img_rows,img_cols))

x_valid = x_valid.astype('float32') / 255.0
x_valid = np.reshape(x_valid, (len(x_valid), 1, img_rows,img_cols))

Y_train = np_utils.to_categorical(y_train, 7)
Y_valid = np_utils.to_categorical(y_valid, 7)
Y_test = np_utils.to_categorical(y_test, 7)

enterface_label=np_utils.to_categorical(enterface_label,7)
savee_label=np_utils.to_categorical(savee_label,7)

print('X_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_img = Input(shape=(1, img_rows,img_cols))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1,3, 3, activation='sigmoid', border_mode='same')(x)

##classifier

y=Flatten()(encoded)
y=Dense(128,activation='relu')(y)
classifier=Dense(7,activation='softmax')(y)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
"""autoencoder.fit(x_train, x_train1,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
autoencoder.fit(x_train, x_train2,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
autoencoder.fit(x_train, x_train3,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
autoencoder.fit(x_train, x_train4,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test))"""
autoencoder.fit(enterface_train, enterface_train,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
autoencoder.fit(savee_train, savee_train,
                nb_epoch=auto_epoch,
                batch_size=batches,
                shuffle=True,
                validation_data=(x_test, x_test)
                )
#autoencoder.save_weights('weight_file.h5',overwrite=True)
                
decoded_imgs = autoencoder.predict(x_test)

"""n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    print str(i)+" pic"
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(img_rows,img_cols))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(img_rows,img_cols))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()"""

classification=Model(input_img,output=classifier)
classification.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
classification.fit(x_train,Y_train,
                        nb_epoch=classify_epoch,
                        batch_size=batches,
                         verbose=1,
                         shuffle=True,
                         validation_data=(x_valid, Y_valid))

classification.fit(enterface_train,enterface_label,
                        nb_epoch=classify_epoch,
                        batch_size=batches,
                         verbose=1,
                         shuffle=True,
                         validation_data=(x_valid, Y_valid))
classification.fit(savee_train,savee_label,
                        nb_epoch=classify_epoch,
                        batch_size=batches,
                         verbose=1,
                         shuffle=True,
                         validation_data=(x_valid, Y_valid))

score = classification.evaluate(x_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
