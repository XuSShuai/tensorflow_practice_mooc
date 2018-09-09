import tensorflow as tf
from PIL import Image
import numpy as np
import mnist_forward
import mnist_backward

def restore_model(img_arr):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        pred = tf.argmax(y, 1)
        
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVEING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)  
                pred_value = sess.run(pred, feed_dict={x:img_arr})
                return pred_value
            else:
                print("model checkpoint file is not found!")
                return -1


def pre_pic(test_pic_path):
    img = Image.open(test_pic_path)
    re_img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(re_img.convert("L"))
    threadhold = 50
    for i in range(28):
        for j in range(28):
            img_arr[i][j] = 255 - img_arr[i][j]
            if img_arr[i][j] < threadhold:
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 255
    img_arr = img_arr.reshape([1, 28*28])
    img_arr = img_arr.astype(np.float32)
    img_arr = np.multiply(img_arr, 1./255)
    return img_arr


def application():
    n = input("input the number of test pictures:")
    for i in range(int(n)):
        test_pic = input("the path of the test picture %d:" % (i))
        test_pic_ready = pre_pic(test_pic)
        pre = restore_model(test_pic_ready)
        print("the prediction number is", pre)

def mnist_app():
    application()
    
if __name__ == "__main__":
    mnist_app()