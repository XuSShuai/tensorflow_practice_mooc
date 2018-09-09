import tensorflow as tf
import mnist_forward
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_generate

STEPS = 50000
BATCH_SIZE = 32
REGULARIZER = 0.001
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
MOVEING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_SAVE_NAME = "mnist_model"
train_num_examples = 60000

def backward(mnist):
    x = tf.placeholder(tf.float32, shape=[None, mnist_forward.INPUT_NODE])
    y = tf.placeholder(tf.float32, shape=[None, mnist_forward.OUTPUT_NODE])
    
    y_hat = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)
    loss_cem = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=y_hat))
    loss = loss_cem + tf.add_n(tf.get_collection("losses"))
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                               global_step, 
                                               train_num_examples/BATCH_SIZE, 
                                               LEARNING_RATE_DECAY, 
                                               staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    
    ema = tf.train.ExponentialMovingAverage(MOVEING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")
        
    saver = tf.train.Saver()
    
    img_batch, label_batch = mnist_generate.get_tfrecord(BATCH_SIZE, True)
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # 续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            
        # 使用多线程提高图片和标签的批获取效率，只需要将批获取的操作放在线程协调器的开启和关闭之间。
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for i in range(STEPS):
            xs, ys = sess.run([img_batch, label_batch])
            _, global_step_value, loss_value = sess.run([train_op, global_step, loss], feed_dict={x:xs, y:ys})
            if i % 1000 == 0:
                print("After %5d steps, the loss is %f" % (global_step_value, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME), global_step)
                
        coord.request_stop()
        coord.join(threads)
                
def main():
    mnist = input_data.read_data_sets("../data/", one_hot=True)
    backward(mnist)
    
if __name__ == "__main__":
    main()