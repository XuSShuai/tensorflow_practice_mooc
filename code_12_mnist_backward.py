import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import code_12_mnist_forward

STEPS = 50000
BATCH_SIZE = 200
REGULARIZER = 0.0001
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, shape=(None, code_12_mnist_forward.INPUT_NODE))
    y = tf.placeholder(tf.float32, shape=(None, code_12_mnist_forward.OUTPUT_NODE))
    
    y_hat = code_12_mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=tf.argmax(y, 1)))
    loss_total = loss + tf.add_n(tf.get_collection("losses"))
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                               global_step, 
                                               mnist.train.num_examples / BATCH_SIZE, 
                                               LEARNING_RATE_DECAY, 
                                               staircase=True)
    
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step=global_step)
    
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")
        
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_val, step = sess.run([train_op, loss_total, global_step], feed_dict={x:xs, y:ys})
            if i % 1000 == 0:
                print("After %6d steps, loss is %f" % (step, loss_val))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)
    
if __name__ == "__main__":
    main()