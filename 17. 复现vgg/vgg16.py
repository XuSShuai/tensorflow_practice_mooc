import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]   # rgb mean

class Vgg:
    def __init__(self, vgg_path):
        if not vgg_path is None:
            self.data_dict = np.load(vgg_path, encoding="latin1").item()
    
    def forward(self, rgb_image):
        """
        rgb_image is a tf.placeholder of shape [1, 224, 224, 3]
        """
        print("building model start")
        start_time = time.time()
        rgb_image_scale = rgb_image * 255.0
        red, green, blue = tf.split(rgb_image_scale, 3, 3)
        gbr_image = tf.concat([blue - VGG_MEAN[0], 
                               green - VGG_MEAN[1],
                               red - VGG_MEAN[2]],
                              3)
        
        assert gbr_image.shape == rgb_image.shape
        
        conv1_1 = self.conv_layer(gbr_image, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        
        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
        
        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")
        
        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")
        
        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool5")
        
        fc6 = self.fc_layer(pool5, "fc6")
        fc6_relu = tf.nn.relu(fc6)
        fc7 = self.fc_layer(fc6_relu, "fc7")
        fc7_relu = tf.nn.relu(fc7)
        fc8 = self.fc_layer(fc7_relu, "fc8")
        self.prob = tf.nn.softmax(fc8, name="prob")
        
        end_time = time.time()
        print("model finished. time consuming: ", end_time - start_time)
        
        data_dict = None
        
        
    def conv_layer(self, x, name):
        with tf.variable_scope(name):
            w = tf.constant(self.data_dict[name][0], name="filter")
            b = tf.constant(self.data_dict[name][1], name="bias")
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
            conv_bias = tf.nn.bias_add(conv, b)
            conv_bias_relu = tf.nn.relu(conv_bias)
            return conv_bias_relu
    
    def fc_layer(self, x, name):
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x_reshape = tf.reshape(x, [-1, dim])
            w = tf.constant(self.data_dict[name][0], name="weight")
            b = tf.constant(self.data_dict[name][1], name="bias")
            return tf.nn.bias_add(tf.matmul(x_reshape, w), b)