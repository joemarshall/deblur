import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp 

SUBPLOT_R=2
SUBPLOT_C=3
SUBPLOT_N=0

def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tfp.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                                  vals,
                                  vals)
    gauss_kernel=tf.expand_dims(gauss_kernel,-1)                                  
    gauss_kernel=tf.expand_dims(gauss_kernel,-1)                                  
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def gblur(img,kernel_size):
    if len(img.shape)==3:
        img=tf.expand_dims(img,0)
    kernel=gaussian_kernel(kernel_size,0,kernel_size/4)
    return tf.nn.conv2d(img,kernel,1,padding='SAME')

def addPlot(im,normalise=False,vmin=0,vmax=1):
    global SUBPLOT_N
    SUBPLOT_N+=1
    if SUBPLOT_N<=SUBPLOT_R*SUBPLOT_C:
    #    im=tf.image.central_crop(im,0.9)
        plt.subplot(SUBPLOT_R,SUBPLOT_C,SUBPLOT_N)
        if len(im.shape)>3:
            # first remove any 1 dimensions at the start
            while im.shape[0]==1 and len(im.shape)>3:
                im=tf.reshape(im,im.shape[1:])
            # then any 1 dimensions at end
            while im.shape[-1]==1 and len(im.shape)>3:
                im=tf.reshape(im,im.shape[:-1])
        if normalise:
            plt.imshow(im,cmap='gray')    
        else:
            plt.imshow(im,vmin=vmin,vmax=vmax,cmap='gray')    


raw = tf.io.read_file("eagle.jpg")
image = tf.io.decode_jpeg(raw, channels=1)
image=tf.cast(image,dtype=tf.float32)
image=image*(1.0/256.0)
unblur_image=image
image=gblur(image,8)

plt.figure()
addPlot(image)

class KernelInit(keras.initializers.Initializer):
    def __init__(self,kernel):
        self.kernel=kernel
    def __call__(self,shape,dtype=None,**kwargs):
        if shape!=self.kernel.shape:
            print("Bad kernel shape:",shape,self.kernel.shape)
            return None
        return tf.cast(self.kernel,dtype)

def make_fixed_model(shape):
    in_blurred=keras.Input(shape=shape)
    kernel=gaussian_kernel(8,0,2)

    deblurrer=layers.Conv2D(1,16,padding='same',activation='tanh')(in_blurred)
    deblurrer2=layers.Conv2D(1,8,padding='same',activation='tanh')(deblurrer)
    deblurrer3=layers.Conv2D(1,4,padding='same',activation='tanh')(deblurrer2)
    deblurrer4=layers.Conv2D(1,2,padding='same',activation='tanh')(deblurrer3)
    out=layers.Conv2D(1,1,activation='tanh')(deblurrer4)
    reblurrer=layers.Conv2D(1,kernel.shape[:-2],name='reblurrer',padding='same',activation='linear',use_bias=False,trainable=False,kernel_initializer=KernelInit(kernel))
    print(reblurrer.weights)
    reblurrerOut=reblurrer(out)
    joined_output=layers.Concatenate()([out,reblurrerOut])
    return keras.Model(inputs=in_blurred,outputs=joined_output),reblurrer

def make_model(shape):
    in_blurred=keras.Input(shape=shape)

    deblurrer=layers.Conv2D(1,16,dilation_rate=1,padding='same',activation='tanh')(in_blurred)
    deblurrer2=layers.Conv2D(1,16,dilation_rate=2,padding='same',activation='tanh')(deblurrer)
    deblurrer3=layers.Conv2D(1,16,dilation_rate=3,padding='same',activation='tanh')(deblurrer2)
    deblurrer4=layers.Conv2D(1,16,dilation_rate=4,padding='same',activation='tanh')(deblurrer3)
    out=layers.Conv2D(1,1,activation='sigmoid')(deblurrer4)
    reblurrer=layers.Conv2D(1,16,name='reblurrer',padding='same',activation='linear',use_bias=False)
    reblurrerOut=reblurrer(out)
    joined_output=layers.Concatenate()([out,reblurrerOut])
    return keras.Model(inputs=in_blurred,outputs=joined_output),reblurrer

def calc_lmg(img):
    if len(img.shape)<4:
        img=tf.expand_dims(img,0)
    dx,dy=tf.image.image_gradients(img)
    dx=dx[:,:-1,:-1,:]
    dy=dy[:,:-1,:-1,:]
    derivatives=tf.stack([dx,dy],axis=-1)
    derivatives=tf.math.sqrt(tf.math.square(derivatives[:,:,:,:,0])+tf.math.square(derivatives[:,:,:,:,1]))
    max_gradients=tf.nn.max_pool2d(derivatives,(16,16),(1,1),padding='SAME')
    max_gradients=tf.image.central_crop(max_gradients,0.9)
    return max_gradients

def zeronorm(img):
    return tf.math.count_nonzero(img,dtype=tf.float32)

def loss_fn(y_true,y_pred):
    global reblurrer
    y_deblurred=y_pred[:,:,:,0:1]
    y_reblurred=y_pred[:,:,:,1:2]
    y_blurred=y_true
    # first loss = how close the two blurred images are
    # i.e. whether deblur and blur functions are good inverses
    lmg_deblurred=calc_lmg(y_deblurred)

    y_blurred=tf.image.central_crop(y_blurred,0.9)
    y_reblurred=tf.image.central_crop(y_reblurred,0.9)
    y_deblurred=tf.image.central_crop(y_deblurred,0.9)

    blur_inverse_loss= tf.reduce_mean((y_reblurred - y_blurred)**2)
    max_gradients=tf.reduce_mean(2-lmg_deblurred)
    kernel_weights=tf.reduce_mean(tf.math.square(reblurrer.weights[0]))
    return blur_inverse_loss#max_gradients+blur_inverse_loss
#    return blur_inverse_loss+10.0*kernel_weights + 0.5*max_gradients

model,reblurrer=make_fixed_model(image.shape[-3:])
model.summary()

lmg=calc_lmg(image)
print(np.min(lmg),np.max(lmg))
addPlot(lmg,vmax=0.5)

model.compile(loss=loss_fn,optimizer=optimizers.Adam())
in_data=np.zeros([1024]+image.shape[1:])
in_data[0]=image[0]
for c in range(1,512):
    in_data[c]=gblur(tf.clip_by_value(tf.random.uniform(shape=image.shape[1:],minval=-4,maxval=1),clip_value_min=0,clip_value_max=1),8)
for c in range(512,1024):
    in_data[c]=gblur(tf.clip_by_value(tf.random.uniform(shape=image.shape[1:],minval=0,maxval=5),clip_value_min=0,clip_value_max=1),8)

model.fit(x=in_data,y=in_data,epochs=1000)

reblur_weights=reblurrer.weights[0]
addPlot(reblur_weights,normalise=True)

output_values= model(image)[0,:,:,:]
out_image=output_values[:,:,0:1]
reblurred_image=output_values[:,:,1:]
addPlot(out_image)

out_lmg=calc_lmg(out_image)

print(tf.reduce_mean(lmg),tf.reduce_mean(out_lmg),tf.math.reduce_min(out_lmg),tf.math.reduce_max(out_lmg))
addPlot(out_lmg,vmax=0.5)
addPlot(reblurred_image)
plt.show()
