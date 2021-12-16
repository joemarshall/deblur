# Implementation of:
# Multi-Scale Neural Network with Dilated
# Convolutions for Image Deblurring
#
# JOSE JAENA MARI OPLE et al. 2020

from typing_extensions import Concatenate
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.activations as activations
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow_probability as tfp 
import threading
import math
import os.path
import sys
from datetime import datetime

from imagegetter import get_image_dataset

tf.keras.mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit("autoclustering") 

IMAGE_BASE_FOLDER="D:\\blurredimages"

BATCH_SIZE=32
INPUT_SHAPE=(256,256)
train_list=os.path.join(IMAGE_BASE_FOLDER,"RealBlur_J_train_list.txt")
test_list=os.path.join(IMAGE_BASE_FOLDER,"RealBlur_J_test_list.txt")

train_ds=get_image_dataset(train_list,crop_size=INPUT_SHAPE)
test_ds=get_image_dataset(test_list,crop_size=INPUT_SHAPE)

class LayerGroup(keras.Model):
    def __init__(self,layer_list,name,**kwargs):
        super().__init__(name=name,**kwargs)
        self.layer_list=layer_list
        for c,l in enumerate(self.layer_list):
            l._name=name+"-%d"%(c+1)

    @tf.function
    def call(self,input):
        for c in self.layer_list:
            input=c(input)
        return input

class reflection_padded_conv2d(keras.Model):
    def __init__(self,filters,dilation_rate=1,stride=1,kernel_size=5):
        super().__init__()
        self.filters,self.dilation_rate,self.stride,self.kernel_size=filters,dilation_rate,stride,kernel_size
        pad_value=(kernel_size-1)//2
        pad_value*=dilation_rate
        self.paddings=tf.constant([[0,0],[pad_value,pad_value],[pad_value,pad_value],[0,0]])
        self.conv=layers.Conv2D(input_shape=(None,None,None),filters=filters,kernel_size=(kernel_size,kernel_size),dilation_rate=(dilation_rate,dilation_rate),strides=(stride,stride),padding='valid',use_bias=True)

    @tf.function
    def call(self,inputs):
        padded=tf.pad(inputs,self.paddings,mode="REFLECT")
        convolved=self.conv(padded)
        return convolved

    def get_config(self):
        return self.filters,self.dilation_rate,self.stride,self.kernel_size


def deconv(filters):
    return layers.Conv2DTranspose(filters,(5,5),strides=(2,2),padding='SAME')

class dilated_conv_block(keras.Model): 
    def __init__(self,filters,**kwargs):
        super().__init__(**kwargs)
        self.filters=filters
        self.dconvs=[]
        for i in range(5):
            self.dconvs.append(reflection_padded_conv2d(filters=filters,dilation_rate=i+1))
        self.concat=layers.Concatenate()
        self.squeezer=layers.Conv2D(filters=filters,kernel_size=(5,5),padding='same')

    @tf.function
    def call(self,input):
        concatted=self.concat([conv(input) for conv in self.dconvs])
        squeezed=self.squeezer(concatted)
        return squeezed

    def get_config(self):
        return self.filters


class res_block(keras.Model):
    def __init__(self,filters):
        super().__init__()
        self.filters=filters
        self.conv1=reflection_padded_conv2d(filters,kernel_size=5)
        self.activation=layers.Activation(activations.relu,input_shape=(None,None,None))
        self.conv2=reflection_padded_conv2d(filters,kernel_size=5)
        self.adder=layers.Add()

    @tf.function
    def call(self,inputs):
        conv=self.conv1(inputs)
        active=self.activation(conv)
        conv2=self.conv2(active)
        return self.adder([conv2,inputs])

    def get_config(self):
        return self.filters

class unet_g_block(keras.Model):
    def __init__(self,name):
        super().__init__(name=name)
#        super().__init__(name=name,dtype=tf.dtypes.float32)
        self.concat=layers.Concatenate()

        self.encoder_1=LayerGroup(
            [
                reflection_padded_conv2d(32,stride=2),
                dilated_conv_block(32),
                res_block(32),
                res_block(32),
                res_block(32),
                res_block(32)
            ],name=name+"ENC_1")
        # E1=w/2,h/2
        self.encoder_2=LayerGroup(
            [
                reflection_padded_conv2d(64,stride=2),
                dilated_conv_block(64),
                res_block(64),
                res_block(64),
                res_block(64),
                res_block(64)
            ],name=name+"ENC_2")
        # E2=w/4,h/4

        self.enc_dec=LayerGroup([
                reflection_padded_conv2d(128,stride=2),
                res_block(128),
                res_block(128),
                res_block(128),
                res_block(128),
                res_block(128),
                deconv(64)
                ],name=name+"ENC_DEC")
        # ED=w/4,h/4

        self.res_2=layers.Add()
        self.dec_2=LayerGroup([
                res_block(64),
                res_block(64),
                res_block(64),
                res_block(64),
                deconv(32)
                ],name=name+"DEC_2")
        # D2=w/2,h/2


        self.res_1=layers.Add()

        self.dec_1=LayerGroup([
                res_block(32),
                res_block(32),
                res_block(32),
                res_block(32),
                res_block(32),
                deconv(3)
                ],name=name+"D1")
        # D1=w,h
    
    @tf.function
    def call(self,inputs):
        concats=self.concat(inputs)
        e_1=self.encoder_1(concats)
        e_2=self.encoder_2(e_1)
        e_d=self.enc_dec(e_2)
        r_2=self.res_2([e_d,e_2])
        d_2=self.dec_2(r_2)
        r_1=self.res_1([d_2,e_1])
        d_1=self.dec_1(r_1)
#           return activations.hard_sigmoid(d_1)
        return d_1

class UnetMultilayerCollection(keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.downsampler=layers.AveragePooling2D()
        self.upsampler=layers.UpSampling2D()
        self.g_block=unet_g_block(name="U")

    @tf.function
    def call(self,input,training=True):
        last_blurred=input
        blurries=[last_blurred]
        for c in range(2):
            last_blurred=self.downsampler(last_blurred)
            blurries.append(last_blurred)

        last_out=self.g_block([last_blurred,blurries.pop()])
        outputs=[last_out]
        for c in range(2):
            last_out=self.upsampler(last_out)
            blurred=blurries.pop()
            last_out=self.g_block([last_out,blurred])
            outputs.append(last_out)
        return outputs

def make_model(input_shape):
#    input=tf.Tensor(shape=input_shape)
    the_layer=UnetMultilayerCollection()
    return the_layer
#    layer_out=the_layer(input)

#    return keras.Model(inputs=input,outputs=layer_out)


@tf.function
def reduce_loss(y_true,y_pred):
    # this loss is done in fp32 otherwise bad things happen
    diff=tf.cast(y_true,tf.float32)-tf.cast(y_pred,tf.float32)
    sq=tf.math.square(diff)
    val= tf.math.reduce_mean(sq,axis=[1,2,3])
    val=tf.sqrt(val)
    val=tf.math.reduce_mean(val)
    return val

@tf.function
def loss1(y_true,y_pred):
    return reduce_loss(y_true,y_pred)

@tf.function
def loss2(y_true,y_pred):
    y_true=tf.nn.avg_pool(y_true,ksize=2,strides=2,padding='SAME')        
    return reduce_loss(y_true,y_pred)*4

@tf.function
def loss4(y_true,y_pred):
    y_true=tf.nn.avg_pool(y_true,ksize=4,strides=4,padding='SAME')        
    return reduce_loss(y_true,y_pred)*16


model=make_model((256,256,3))
#model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.9,beta_2=0.999,epsilon=10e-8),loss=[loss4,loss2,loss1])

class PlotCallback(keras.callbacks.Callback):
    def __init__(self,train_data,test_data):
        self.plots={}
        for x,y in train_data:
            self.train_data=x,y
        for x,y in test_data:
            self.test_data=x,y
        print(self.train_data[0].shape)
        self.on_epoch_end(-1)

    def addPlot(self,im,r,c,n,normalise=False,vmin=0,vmax=1):
        if len(im.shape)>3:
            # first remove any 1 dimensions at the start
            while im.shape[0]==1 and len(im.shape)>3:
                im=tf.reshape(im,im.shape[1:])
            # then any 1 dimensions at end
            while im.shape[-1]==1 and len(im.shape)>3:
                im=tf.reshape(im,im.shape[:-1])
        im=tf.cast(im,dtype=tf.dtypes.float32)
        im=tf.clip_by_value(im,0.0,1.0)
        if not (r,c,n) in self.plots:
            plt.subplot(r,c,n)
            if normalise:
                self.plots[(r,c,n)]=plt.imshow(im,cmap='gray')    
            else:
                self.plots[(r,c,n)]=plt.imshow(im,vmin=vmin,vmax=vmax,cmap='gray')
        else:
            self.plots[(r,c,n)].set_data(im)
        plt.draw()

    def on_epoch_end(self,epoch,logs=None):
        output_train= model.predict(tf.expand_dims(self.train_data[0],0))
        output_test = model.predict(tf.expand_dims(self.test_data[0],0))

        self.addPlot(self.train_data[0],2,5,1)
        self.addPlot(output_train[0],2,5,2) 
        self.addPlot(output_train[1],2,5,3)
        self.addPlot(output_train[2],2,5,4)
        self.addPlot(self.train_data[1],2,5,5)
        self.addPlot(self.test_data[0],2,5,6)
        self.addPlot(output_test[0],2,5,7) 
        self.addPlot(output_test[1],2,5,8)
        self.addPlot(output_test[2],2,5,9)
        self.addPlot(self.test_data[1],2,5,10)


def fit_thread():    
    # find last saved weights and load them if possible
    start_epoch=0
    models=os.listdir("models")
    if len(models)>0:
        lastModel=sorted(models)[-1]
        print("Loading last model:",lastModel)
        root,ext=os.path.splitext(os.path.basename(lastModel))
        start_epoch=int(root.split("-")[-1])
        
        # call model so weights get created
        model.build(train_ds.batch(BATCH_SIZE).take(1).get_single_element()[0].shape)
        model.load_weights(os.path.join("models",lastModel))
    logdir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path="models/" + datetime.now().strftime("%Y%m%d-%H%M%S")+"-{epoch:02d}.hdf5"
    model.fit(train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE),epochs=1000,initial_epoch=start_epoch,validation_freq=1,callbacks=[PlotCallback(train_ds.take(1),test_ds.take(1)),tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=10),tf.keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)])

plt.figure()
threading.Thread(target=fit_thread,daemon=True).start()
plt.show()

