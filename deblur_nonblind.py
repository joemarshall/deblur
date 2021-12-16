import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp 
import threading
import math

SUBPLOT_R=2
SUBPLOT_C=1
SUBPLOT_N=0
PATCH_SIZE=64
BATCH_SIZE=64


def getImage(filename):
    raw = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(raw, channels=1)
    image=tf.cast(image,dtype=tf.float32)
    return image*(1.0/256)

class PatchSequence(keras.utils.Sequence):
    def __init__(self,img,gt,batch_size):        
        self.img=img.numpy()
        self.gt=gt.numpy()
        w=self.img.shape[1]
        h=self.img.shape[0]
        half_size=int(PATCH_SIZE//2)
        self.img_padded=np.pad(self.img,((half_size,half_size),(half_size,half_size),(0,0)),mode='constant')
        self.batch_size=batch_size
        self.length=math.ceil((w*h)/batch_size)

    def _get_patch(self,pos):
        half_size=PATCH_SIZE//2
        w,h=self.img.shape[1],self.img.shape[0]
        x=(pos % w) + half_size
        y=(pos // w) + half_size
        x1=x-half_size
        x2=x+half_size
        y1=y-half_size
        y2=y+half_size
        patch=self.img_padded[y1:y2,x1:x2,:]
        return patch

    def _get_pos_range(self,pos1,pos2):
        return [(pos % self.img.shape[1],pos // self.img.shape[1]) for pos in range(pos1,pos2)]

    def __getitem__(self,n):
        start_pos=n*self.batch_size
        end_pos=min(start_pos+self.batch_size,self.img.shape[-3]*self.img.shape[-2])        
        batch_x=np.stack([self._get_patch(c) for c in range(start_pos,end_pos)],axis=0)
        batch_y= [self.gt[y,x,:] for x,y in self._get_pos_range(start_pos,end_pos)]
        batch_y=np.stack(batch_y,axis=0)
        return batch_x,batch_y

    def __len__(self):
        return self.length


def make_patches(img,gt):
    return PatchSequence(img,gt,BATCH_SIZE)


class PlotCallback(keras.callbacks.Callback):
    def __init__(self):
        self.plots={}

    def addPlot(self,im,r,c,n,normalise=False,vmin=0,vmax=1):
        if len(im.shape)>3:
            # first remove any 1 dimensions at the start
            while im.shape[0]==1 and len(im.shape)>3:
                im=tf.reshape(im,im.shape[1:])
            # then any 1 dimensions at end
            while im.shape[-1]==1 and len(im.shape)>3:
                im=tf.reshape(im,im.shape[:-1])
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
        output_values= model.predict(x)
        out_image=tf.reshape(output_values,gt.shape)
        self.addPlot(out_image,2,1,1)
        patchTest=x._get_patch(120*400+100)
        print(patchTest.shape)
        self.addPlot(patchTest,2,1,2)

def make_model(channels):
    in_blurred=keras.Input(shape=(PATCH_SIZE,PATCH_SIZE,channels))
    last_layer_out=in_blurred
#    reshaped=layers.Reshape((PATCH_SIZE*PATCH_SIZE*channels,))(last_layer_out)
#    last_layer_out=reshaped
 
    last_layer_out=layers.Conv2D(128,16,activation='relu',padding='same')(last_layer_out)
    last_layer_out=layers.Conv2D(128,8,activation='relu',padding='same')(last_layer_out)
    last_layer_out=layers.Conv2D(128,4,activation='relu',padding='same')(last_layer_out)
    last_layer_out=layers.Conv2D(128,2,activation='relu',padding='same')(last_layer_out)
 
#     conv_size=5
#     filters=128
#     for c in range(6):
#         prev_out=last_layer_out
#         last_layer_out=layers.Conv2D(filters,(conv_size,conv_size),strides=(2,2),activation='relu',padding='same')(last_layer_out)
#         pooled=layers.MaxPooling2D()(prev_out)
#         last_layer_out=layers.Concatenate()([last_layer_out,pooled])
# #        last_layer_out=layers.Dense(32)(last_layer_out)
    last_layer_out=layers.Flatten()(last_layer_out)
    out_deblurred=layers.Dense(1,activation='sigmoid')(last_layer_out)
    return keras.Model(inputs=in_blurred,outputs=out_deblurred)

#def loss_fn(y_true,y_pred):
#    return keras.losses.Lms


gt = getImage("eagle.jpg")
blurred = getImage("eagle_blurred.jpg")
print(blurred.shape,gt.shape)

model=make_model(gt.shape[-1])
model.summary()


x=make_patches(blurred,gt)
print("Made x,y")

model.compile(loss='mse',optimizer=optimizers.Adam())
print("Compiled model, now fitting - ")

def fit_thread():
    model.fit(x,epochs=1000,steps_per_epoch=None,callbacks=[PlotCallback()])

threading.Thread(target=fit_thread,daemon=True).start()
plt.figure()
plt.show()