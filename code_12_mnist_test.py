import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import code_12_mnist_forward
import code_12_mnist_backward

TEST_INTERVEL_SEC = 10

def test(mnist):
    with tf.Graph().as_default() as g:  # 使用tf.Graph()复现计算图
        x = tf.placeholder(tf.float32, [None, code_12_mnist_forward.INPUT_NODE])
        y = tf.placeholder(tf.float32, [None, code_12_mnist_forward.OUTPUT_NODE])
        y_hat = code_12_mnist_forward.forward(x, None)
        
        ema = tf.train.ExponentialMovingAverage(code_12_mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(code_12_mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path) # 恢复模型到当前会话
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    accuracy_val = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
                    print("After training %s steps, test accuracy is %f" % (global_step, accuracy_val))
                else:
                    print("No checkpoint file find")
                    return
            time.sleep(TEST_INTERVEL_SEC)
            
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)
    
if __name__ == "__main__":
    main()