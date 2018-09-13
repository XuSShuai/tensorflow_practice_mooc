import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from Nclasses import labels
import vgg16
import utils
import numpy as np

img_path = input("input the path and image name: ")
img_ready = utils.load_image(img_path)
print("image shape: ", img_ready.shape)

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
    vgg = vgg16.Vgg(vgg_path="../vgg/vgg16.npy")
    vgg.forward(x)
    probability = sess.run(vgg.prob, feed_dict={x:img_ready})
    top_5 = np.argsort(probability[0])[-1:-6:-1]
    bar_value = []
    bar_name = []
    for i in top_5:
        print("prob: %8f ------ %s " % (probability[0][i], ", ".join(labels[i].split("\n"))))
        bar_value.append(probability[0][i])
        bar_name.append(labels[i])
        
    grid = plt.GridSpec(2, 3)
    plt.subplot(grid[0:2,0:2])
    plt.imshow(Image.open(img_path))
    plt.subplot(grid[:,2:3])
    plt.barh(range(len(bar_value)), bar_value, tick_label=bar_name, fc='g', height=0.2)
    for x, y in zip(range(len(bar_value)), bar_value):
        plt.text(y + 0.005, x, y)
        
    plt.savefig("./result.png")