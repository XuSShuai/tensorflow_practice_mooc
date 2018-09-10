import tensorflow as tf
import lenet_5_mnist_forward
import lenet_5_mnist_backward
from PIL import Image
import numpy as np


def restore_model(img_ready):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, 
                           shape=[1, 
                                  lenet_5_mnist_forward.IMAGE_SIZE, 
                                  lenet_5_mnist_forward.IMAGE_SIZE, 
                                  lenet_5_mnist_forward.NUM_CHANNELS])
        y = lenet_5_mnist_forward.forward(x, False, None)
        prob = tf.argmax(y, 1)
        
        ema = tf.train.ExponentialMovingAverage(lenet_5_mnist_backward.MOVEING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(lenet_5_mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                prob_value = sess.run(prob, feed_dict={x:img_ready})
                return prob_value
            else:
                print("model checkpoint file is not found")
                return
            

def pre_pic(img_path):
    img = Image.open(img_path)
    img = img.resize((lenet_5_mnist_forward.IMAGE_SIZE, lenet_5_mnist_forward.IMAGE_SIZE), Image.ANTIALIAS)
    img_arr = np.array(img.convert("L"))
    img_arr = img_arr[np.newaxis,:,:,np.newaxis]
    img_ready = 255 - img_arr
    img_ready = img_ready.astype(np.float32)
    img_ready = np.multiply(img_ready, 1./255)
    return img_ready


def application():
    n = input("input the number of test image:")
    for i in range(int(n)):
        img_path = input("the path of the test image(%s)" % i)
        img_ready = pre_pic(img_path)
        pred = restore_model(img_ready)
        print("the prediction number is ", pred)
        
        
if __name__ == "__main__":
    application()