import tensorflow as tf
import os.path

from tensorflow.python.data.ops.dataset_ops import Dataset

def get_image_dataset(image_list,crop_size=None,crop_offset=None):
    image_folder=os.path.dirname(image_list)
    def process_image_pairs(images,randoms):
        blurred = tf.io.read_file(images[0])
        gt = tf.io.read_file(images[1])
        blurred =  tf.io.decode_png(blurred, channels=3)
        gt =  tf.io.decode_png(gt, channels=3)
        if crop_size!=None:
            w,h=crop_size
            if crop_offset!=None:
                x,y=crop_offset
            else:
                x=int(randoms[1]*(tf.cast(tf.shape(gt),tf.dtypes.float32)[1]-w))
                y=int(randoms[0]*(tf.cast(tf.shape(gt),tf.dtypes.float32)[0]-h))
            blurred=blurred[y:y+w,x:x+w,:]
            gt=gt[y:y+w,x:x+w,:]
        blurred=tf.cast(blurred,float)*(1.0/256)
        gt=tf.cast(gt,float)*(1.0/256)
        return blurred,gt

    tf.random.set_seed(17)
    paths=[]
    for line in open(image_list,'r'):
        s=line.split(' ')
        if len(s)>=2:
            ground_truth=os.path.join(image_folder,s[0])
            blurred=os.path.join(image_folder,s[1])
            paths.append([blurred,ground_truth])
    ds=tf.data.Dataset.from_tensor_slices(paths)
    randoms=tf.data.Dataset.from_tensor_slices(tf.random.uniform((len(paths),2)))
    ds=tf.data.Dataset.zip((ds,randoms))
    ds=ds.map(process_image_pairs, num_parallel_calls=4)
    return ds

    


if __name__=="__main__":
    import matplotlib.pyplot as plt
    IMAGE_BASE_FOLDER="D:\\blurredimages"
    train_list=os.path.join(IMAGE_BASE_FOLDER,"RealBlur_J_train_list.txt")
    train_ds=get_image_dataset(train_list,crop_size=(256,256))
    for x,y in train_ds.take(1):
        print(x.shape,y.shape)
        plt.subplot(2,1,1)
        plt.imshow(x)
        plt.subplot(2,1,2)
        plt.imshow(y)
    plt.show()
