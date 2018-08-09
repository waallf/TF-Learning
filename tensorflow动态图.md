## tensorflow 动态图（Eager Execution--挨个执行）

[参考](https://blog.csdn.net/uwr44uouqcnsuqb60zk2/article/details/78431019)

[官方readme](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/README.md )

[eager execution 用户指南 ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/g3doc/guide.md )

Eager Execution 是一个运行定义的   **接口**，一旦被Python调用，就立即执行



当启动Eager Execution时，运算会立刻执行，无需Session.run()，就可以返回值

* 调用

  `tfe.enable_eager_execution()`启动eager

  ```
  import tensorflow.contrib.eager as tfe
  tfe.enable_eager_execution()
  
  ```

  

* 动态模型的构建可使用Python控制流

  

  ```
  import tensorflow.contrib.eager as tfe
  tfe.enable_eager_execution()
  
  a = tf.constant(12)
  counter = 0
  while not tf.equal(a, 1):
   if tf.equal(a % 2, 0):
     a = a / 2
   else:
     a = 3 * a + 1
   print(a)
  ```

  上面 tf.constant(12）张量对象的使用将把所有数学运算提升为张量运算，从而所有的返回值将是张量。 

  

* 梯度

  **tfe.gradients_function( function)** 参数是一个函数，对这个函数进行求导，然后将值带入

  自动求导，与pytorch的 autograd 类似  

  ```
  def squre(x):
  	return tf.multiply(x,x)
  grad = tfe.gradients_function(squre)
  print(squre(3.))
  print(grad(3.))
  # 求二阶倒数
  gradgrad = tfe.gradients_function(lambda x: grad(x)[0])
  print(gradgrad(3.)) 
  ```

  自定义梯度:

  ​	

  ```
  def log1pexp(x):
   e = tf.exp(x)
   def grad(dy):
   return dy * (1 -1/(1+e))
  return tf.log(1+e),grad
  grad_log1pexp = tfe.gradients_function(log1pexp)
  # Gradient at x = 0 works as before.
  print(grad_log1pexp(0.))
  # [0.5]
  # And now gradient computation at x=100 works as well.
  print(grad_log1pexp(100.))
  # [1.0]
  ```

* 建立模型

  创建一个简单的两层网络对标准的MNIS数字进行分类

  ```
  class MNISTModel(tfe.Network):
   def __init__(self):
     super(MNISTModel, self).__init__()
     self.layer1 = self.track_layer(tf.layers.Dense(units=10))
     self.layer2 = self.track_layer(tf.layers.Dense(units=10))
   def call(self, input):
     """Actually runs the model."""
     result = self.layer1(input)
     result = self.layer2(result)
     return result
     
  model = MNISTModel()
  batch = tf.zeros([1, 1, 784])
  print(batch.shape)
  # (1, 1, 784)
  result = model(batch)
  print(result)
  ```

  **这里没有使用占位符或者会话，一旦数据输入，层的参数就被设定好了。**

  ```
  def loss_function(model, x, y):
   y_ = model(x)
   return tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
   
  # 开始训练
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
  for (x, y) in tfe.Iterator(dataset):
   grads = tfe.implicit_gradients(loss_function)(model, x, y)
   optimizer.apply_gradients(grads)
  ```

  **implicit_gradients**计算损失函数  关于所有变量的倒数 

  运用GPU训练  

  ```
  with tf.device("/gpu:0"):
   for (x, y) in tfe.Iterator(dataset):
     optimizer.minimize(lambda: loss_function(model, x, y))
  ```

  ***

  optimizer.minimize  与 apply_gradients()  等价

  

## Eager 与 Graphapply_gradients()  

Eager execution 使开发和调试互动性更强，但是 TensorFlow graph 在分布式训练、性能优化和生产部署中也有很多优势。

启动 Eager execution时，执行运算的代码还可以构建一个 描述Eager execution **未启用时的计算图**。 为了将模型转化为Graph，只需要在 eager execution 未启用的 Python session 中运行同样的代码 。[事例](./eager_execution_demo.py)

```
# graph

x = tf.placeholder(tf.float.32,shape[1,1])
m = tf.matmul(x,x)
with tf.Session() as sess:
  print(sess.run(m, feed_dict={x: [[2.]]}))

# Will print [[4.]]
```

```
# eager execution 

x = [[2.]]
m = tf.matmul(x, x)

print(m)
```



---

目前只有少量针对Eager 的API，大多数的API与运算都需要和启动的eager一起工作。

**注意** 

1. 读取数据时，不能用排队切换，要使用tf.data  [教程1](https://developers.googleblog.com/2017/09/introducing-tensorflow-datasets.html ) [教程2](https://www.tensorflow.org/programmers_guide/datasets )
2. 使用目标导向的层（比如 tf.layer.Conv2D() 或者 Keras 层 ），他们可以**直接存储变量**，你可以为大多数模型写代码，这对 eager execution 和图构建同样有效。也有一些例外，比如动态模型使用 Python 控制流改变基于输入的计算。一旦调用 tfe.enable_eager_execution()，它不可被关掉。为了获得图行为，需要建立一个新的 Python session。 

## 开始使用 



* [安装Tensorflow的nightly版本](https://github.com/tensorflow/tensorflow#installation )
* 