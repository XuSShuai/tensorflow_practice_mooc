# tensorflow_practice_mooc
tensorflow practice mooc

## 1 - 前向传播

```python
def forward(x, regularizer):
    w = 
    b = 
    y = 
    return y
    
def get_weight(shape, regularizer):
    w = tf.Variable()
    tf.add_to_collection("loss", ...)
    return w

def get_bias(shape):
    b = 
    return b
```


## 2 - 反向传播

```python
def backward():
    x = tf.placeholder()
    y = tf.placeholder()
    y_hat = 
    global_step = tf.Variable(0, trainable=False)
    loss = 
    <正则化，指数衰减学习率，滑动平均>
    train = 
    实例化saver
    with tf.Session() as sess:
        初始化所有变量
        for i in range(STEPS):
            sess.run(train, feed_dict={x:, y:})
            if i % 轮数 == 0:
                print()
                saver.save()
```

### 2.1 - 正则化
#### 2.1.1 - 反向传播
```python
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=tf.argmax(y, 1))
cem = tf.reduce_mean(ce)
loss = cem + tf.add_n(tf.get_collection("losses"))
```
#### 2.1.2 - 前向传播
```python
if regularizer != None:
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
```
### 2.2 - 指数衰减学习率
```python
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, 
                                            global_step, 
                                            LEARNING_RATE_STEP, 
                                            LEARNING_RATE_DECAY, 
                                            staircase = True)
```
### 2.3 - 滑动平均
```python
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
ema_op = ema.apply(tf.trainable_variables())
with tf.control_dependencies([train_step, ema_op]):
    train_op = tf.no_op(name="train")
```

## 3 - 测试模型

```python
def test(mnist):
    with tf.Graph().as_default() as g:  # tf.Graph()加载计算图中的节点
        定义x，y，y_hat
        实例化可以计算滑动平均值的saver对象
        正确率运算图定义
        while True:
            with tf.Session() as sess:
                不再初始化所有变量，而是加载ckpt模型ckpt = tf.train.get_checkpoint_path(存储路径)
                如果已经有了ckpt模型：if ckpt and ckpt.model_checkpoint_path:
                    恢复模型到当前会话 saver.restore(sess, ckpt.model_checkpoint_path)
                    恢复轮数 global_step =
                    执行正确率的计算 accuracy_score = 
                    打印
                如果没有模型：
                    提示模型没有找到
```
