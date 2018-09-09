import mnist_backward
import mnist_forward
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import mnist_generate

TEST_INTERVEL_SEC = 10
test_num_examples = 10000

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        
        y_hat = mnist_forward.forward(x, None)
        
        correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVEING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        
        img_batch, label_batch = mnist_generate.get_tfrecord(test_num_examples, False)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    
                    coord = tf.train.Coordinator()
                    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
                    
                    xs, ys = sess.run([img_batch, label_batch])
                    accuracy_value = sess.run(accuracy, feed_dict={x:xs, y:ys})
                    
                    coord.request_stop()
                    coord.join(threads)
                    
                    print("After training %s steps, the accuracy in test set is %f" % (global_step, accuracy_value))
                else:
                    print("model checkpoint path is not found")
                    return
            time.sleep(TEST_INTERVEL_SEC)
            
def main():
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    test(mnist)
    
if __name__ == "__main__":
    main()