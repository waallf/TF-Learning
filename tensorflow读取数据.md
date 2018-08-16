

[参考](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/reading_data.html)

## TF的数据读取方式：

1. 供给数据(Feeding)  :  在TensorFlow程序运行的每一步， 让Python代码来供给数据 
2. 从文件读取数据： 在TensorFlow图的起始， 让一个输入管线从文件中读取数据 
3. 预加载数据 ：在TensorFlow图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况)。 



###数据供给

TF的数据供给机制允许将数据直接输入到张量中，因此python运算可以直接把数据设置到运算图中。

**通过`tf.placrholder`将数据直接输入到tensorflow的计算图中**

```
with tf.Session():
	input = tf.placeholder(tf.float32)
	classifier = ...
	print(classifier.eval(feed_dict{input:data}))
```



## 从文件读取数据

一共典型的文件读取管线会包含下面这些步骤：

1. 文件名列表
2. *可配置的* 文件名乱序(shuffling)
3. *可配置的* 最大训练迭代数(epoch limit)
4. 文件名队列
5. 针对输入文件格式的阅读器
6. 纪录解析器
7. *可配置的*预处理器
8. 样本队列

![AnimatedFileQueues](./assets/AnimatedFileQueues.gif)

### 文件名，乱序，最大迭代数 

使用`tf.train.match_filenames_once`或者`[文件名1，。。。。]`来产生文件名列表

将产生的文件名交给`tf.train.string_input_producer`,生产一个先入先出的队列，文件阅读器会需要他来读取数据。`string_input_producer`提供参数来设置文件名乱序和最大迭代次数，`QunnerRunner`会在每次迭代时将所有的文件名加入队列中。如果shuffle=ture的话，会对文件名进行乱序处理。

`QueneRunner`的工作线程是独立于文件阅读器的线程， 因此乱序和将文件名推入到文件名队列这些过程不会阻塞文件阅读器运行。 

### 文件格式

不同的文件选取不同的与阅读器，然后将文件名队列提供给阅读器的`read`方法。该方法会输出一个ley，来表征输入的文件和其中的记录，同时得到一个字符串标量**（就是数据）**，这个字符串标量可以被一个或多个解析器，或者转换操作将其解码为张量并且构造成为样本。 

### csv文件

阅读器：tf.TextLineReader()

解析器：tf.decode_csv()

```
# 创建文件名列表,产生队列
filename_queue = tf.train.string_input_producer(["file1.csv","file2.csv"])

# 创建阅读器
reader = tf.TextLineReader()

# 使用阅读器的读方法
key,value = reader.read(file_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]

# 解码数据

col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
    
features = tf.concat(0, [col1, col2, col3, col4])
with tf.Session() as sess:
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	for i in range(1200):
        # Retrieve a single instance:
        example, label = sess.run([features, col5]
     coord.request_stop()
     coord.join(threads)
```

从二进制文件中读取固定长度数据： 阅读器 tf.FixedLengthRecorderReader(),解析器tf.decode_raw()其将一个字符串转化为unit8张量。[例子 CIFAR-10读取](https://www.cnblogs.com/lixiaoran/p/6740022.html)

---

在调用`run`之前，必须调用tf.train.start_queue_runners将文件名填充到队列，否则`read`操作将会等待，直到队列中有值为止

---



## TFrecord格式



## tfrecord介绍：  

`tfrecord`包含了`tf.train.Example`协议缓冲区，先获取数据，然后将数据填 入到Example协议缓冲区，将协议缓冲区序列化为一个字符串，然后通过`tf.python_io.TFRecordWriter`写入`TFRecoed  `

阅读器：`tf.TFRecordReader`

解析器：`tf.parse_single_example` ,将协议缓冲区内的数据解析为张量。

#### 写入tfrecord文件

```
tfrecord_filename = `./tfrecords/train.tfrecords`
writer = tf.python_io.TFRecordWriter(tfrecords_filename)
for (读取图片,label)：
  example = tf.train.Example(features = tf.train.Features(
            feature={
            "label":tf.train.Feature(init_list = tf.train.Int64List(value = label)),
            "image":tf.train.Feature(bytes_list = tf.train.BytesList(value = [image]))}))
  writer.write(example.SerializeToString())
writer.close()
```

#### 读取tfrecord

从TFRecord文件中读取数据，需要首先使用tf.train.string_input_producer生成一个解析队列  
然后使用tf.TFRecordReader读取生成的解析队列  在使用tf.paese_single_example()解析器来解析队列  
```
# 产生文件名列表
tfrecords_filename = "train.tfrecords" 
#生成解析队列
file_queue = tf.train.string_input_producer([tfrecords_filenmae],)
#创建阅读器
reader = tf.TFRecordReader()
_,serialiezd_example = reader.read(file_queue)
#解析器进行解析
feature = tf.parse_single_example(serialiezd_example,
                                  features={
                                  'label':tf.FixedLenFeature([],tf.int64),
                                  "ima_raw":tf.FixedLenFeature([],tf.string),
  
  })
  
img = feature['label']
label = feature['ima_raw']

                                    
```
*注意以下两句*  
```
with tf.Session() as sess:
  coord=tf.train.Coordinator()
  threads= tf.train.start_queue_runners(coord=coord)
  for i in range(20):
    example, l = sess.run([image,label])#必须开启了队列才可以读取
  coord.request_stop()
  coord.join(threads)
```

---

---

#### 将读取到的数据以batch的方式送入网络训练 

```
def read_my_file_format(filename_queue):
  reader = tf.SomeReader()
  key, record_string = reader.read(filename_queue)
  example, label = tf.some_decoder(record_string)
  processed_example = some_processing(example)
  return processed_example, label

def input_pipeline(filenames, batch_size, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch
```

***如果需要对不同文件中的数据有更强的乱序以及并行处理，可以使用`tf.train.shuffle_batch_join`**

```
def read_my_file_format(filename_queue):
  # Same as above

def input_pipeline(filenames, batch_size, read_threads, num_epochs=None):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example_list = [read_my_file_format(filename_queue)
                  for _ in range(read_threads)]
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch_join(
      example_list, batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch
```

在这个例子中， 你虽然只使用了一个文件名队列， 但是TensorFlow依然能保证多个文件阅读器从同一次迭代(epoch)的不同文件中读取数据，知道这次迭代的所有文件都被开始读取为止。（通常来说一个线程来对文件名队列进行填充的效率是足够的）

另一种替代方案是： 使用[`tf.train.shuffle_batch` 函数](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/tensorflow-zh/SOURCE/api_docs/python/io_ops.html),设置`num_threads`的值大于1。 这种方案可以保证同一时刻只在一个文件中进行读取操作(但是读取速度依然优于单线程)，而不是之前的同时读取多个文件。这种方案的优点是：

- 避免了两个不同的线程从同一个文件中读取同一个样本。
- 避免了过多的磁盘搜索操作。



#### 创建线程并使用QueueRunner来读取

* 在进行训练步骤之前，需要调用`tf.train.start_queue_runners`，它将会启动输入管道线程，填充样本到队列中，以便可以拿到样本。 在调用他时，最好配合使用`tf.train.Coordinator`

* 如果设置了迭代次数  (`tf.train.string_input_producer`中设置`num_epoch`),则可以使用以下代码：

  ```
  # Create the graph, etc.
  init_op = tf.initialize_all_variables()
  
  # Create a session for running operations in the Graph.
  sess = tf.Session()
  
  # Initialize the variables (like the epoch counter).
  sess.run(init_op)
  
  # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess,coord=coord)
  try:
  	while not coord.should_stop():
  		sess.run(train_op)
  except tf.errors.OutOfRangeError:
  	# Create the graph, etc.
  
  ```

  **达到训练步数后怎样清理关闭线程 ** 

  当达到设置的训练步数后，会激发OutOfRange错误，让偶关闭文件名队列，清理线程，关闭队列做了两件事情：

  * 如果还试着对文件名队列执行入队操作时将发生错误 
  * 任何当前或将来出队操作要么成功（队列中还有足够的元素）或立即失败，不会再等待足够的元素加入其中。

  当关闭文件名队列的时候，有可能还有文件在该队列中，那么下一阶段的操作（reader）还会执行一段时间，一旦文件名队列为空以后，再读取数据将会发生`OutOfRange` 错误。在这种情况下，即使你可能有一个QueueRunner关联着多个线程。如果这不是在QueueRunner中的最后那个线程，`OutOfRange`错误仅仅只会使得一个线程退出。这使得其他那些正处理自己的最后一个文件的线程继续运行，直至他们完成为止。 （但如果假设你使用的是[`tf.train.Coordinator`](http://wiki.jikexueyuan.com/project/api_docs/python/train.html)，其他类型的错误将导致所有线程停止）。一旦所有的reader线程触发`OutOfRange`错误，然后才是下一个队列，再是样本队列被关闭。 

  同样，样本队列中会有一些已经入队的元素，所以训练将持续到队列中没有样本为止，。如果样本队列是一个[`RandomShuffleQueue`](http://wiki.jikexueyuan.com/project/api_docs/python/io_ops.html)，因为你使用了`shuffle_batch` 或者 `shuffle_batch_join`，所以通常不会出现以往那种队列中的元素会比`min_after_dequeue` 定义的更少的情况。 然而，一旦该队列被关闭`min_after_dequeue`设置的限定值将失效，最终队列将为空。在这一点来说，当实际训练线程尝试从样本队列中取出数据时，将会触发`OutOfRange`错误，然后训练线程会退出。一旦所有的培训线程完成，[`tf.train.Coordinator.join`](http://wiki.jikexueyuan.com/project/api_docs/python/train.html)会返回，就可以正常退出了。 

  

  ### 筛选记录或每个记录产生多个样本

  举个例子，有形式为`[x, y, z]`的样本，我们可以生成一批形式为`[batch, x, y, z]`的样本。 如果你想滤除这个记录（或许不需要这样的设置），那么可以设置batch的大小为0；但如果你需要每个记录产生多个样本，那么batch的值可以大于1。 然后很简单，只需调用批处理函数（比如： `shuffle_batch` or `shuffle_batch_join`）去设置`enqueue_many=True`就可以实现。（`enqueue_many`用于设置在每一次读取数据时，每个样例是否只读入一次）

  

  ### 预加载数据

  `tf.placeholder` 

  ### 多管道输入

  就是再训练的同时也读入了验证集数据，进行验证

  ---

  # 另一种速记读取方法

  

  ### `tf.data.TFRecordDataset`

  [参考](https://blog.csdn.net/yeqiustu/article/details/79793454) 

  

  1.从tfrecord文件创建 TFRecordDataset：

  ​	`dataset = tf.data.TFRecordDataset('xxx.tfrecord')`

  2.解析数据：

  * 解析单个数据：

    ```
    def parse_exmp(serial_exmp):  	
    	feats = tf.parse_single_example(serial_exmp, features=	 {'feature':tf.FixedLenFeature([], tf.string),\
    	'label':tf.FixedLenFeature([10],tf.float32), 'shape':tf.FixedLenFeature([x], tf.int64)})
    	image = tf.decode_raw(feats['feature'], tf.float32)
    	label = feats['label']
    	shape = tf.cast(feats['shape'], tf.int32)
    	return image, label, shape
    ```

  * 解析tfrecord中全部数据，使用dataset的map方法：

    ```
    dataset = dataset.map(parse_exmp) 
    dataset = dataset.repeat(epochs).shuffle(buffer_size).batch(batch_size)
    ```

  * 解析完数据后，便可以取出数据进行使用，通过创建iterator进行：

    ```
    iterator_train = dataset.make_one_shot_iterator()
    batch_image,batch_label,batch_shape = iterator_train.get_next()
    
    ```

  截止到此，读取数据已经完成，接下来是与此方法配套的placeholder用法

  * 创建`iterator placeholder`

    ```
    handle = tf.placeholder(tf.string,shape[])
    iterator = tf.data.Iterator.from_string_handle(handle,dataset.output_types,dataset.output_shapes)
    image,label,shape = iterator.get_next() #现在要依靠它来获得下一批数据
    
    with tf.Session() as sess:
    	handle_train = sess.run(iter_train.string_handle())
            sess.run([loss, train_op], feed_dict={handle: handle_train}
    ```

    

   汇总：

  ```
  import tensorflow as tf
  
  train_f, val_f, test_f = ['mnist-%s.tfrecord'%i for i in ['train', 'val', 'test']]
  
  def parse_exmp(serial_exmp):
  	feats = tf.parse_single_example(serial_exmp, features={'feature':tf.FixedLenFeature([], tf.string),\
  	'label':tf.FixedLenFeature([10],tf.float32), 'shape':tf.FixedLenFeature([], tf.int64)})
  	image = tf.decode_raw(feats['feature'], tf.float32)
  	label = feats['label']
  	shape = tf.cast(feats['shape'], tf.int32)
  	return image, label, shape
  
  
  def get_dataset(fname):
  	dataset = tf.data.TFRecordDataset(fname)
  	return dataset.map(parse_exmp) # use padded_batch method if padding needed
  
  epochs = 16
  batch_size = 50  # when batch_size can't be divided by nDatas, like 56,
  		# there will be a batch data with nums less than batch_size
  
  # training dataset
  nDatasTrain = 46750
  dataset_train = get_dataset(train_f)
  dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size) # make sure repeat is ahead batch
  			# this is different from dataset.shuffle(1000).batch(batch_size).repeat(epochs)
  			# the latter means that there will be a batch data with nums less than batch_size for each epoch
  			# if when batch_size can't be divided by nDatas.
  nBatchs = nDatasTrain*epochs//batch_size
  
  # evalation dataset
  nDatasVal = 8250
  dataset_val = get_dataset(val_f)
  dataset_val = dataset_val.batch(nDatasVal).repeat(nBatchs//100*2)
  
  # test dataset
  nDatasTest = 10000
  dataset_test = get_dataset(test_f)
  dataset_test = dataset_test.batch(nDatasTest)
  
  # make dataset iterator
  iter_train = dataset_train.make_one_shot_iterator()
  iter_val   = dataset_val.make_one_shot_iterator()
  iter_test   = dataset_test.make_one_shot_iterator()
  
  # make feedable iterator
  handle = tf.placeholder(tf.string, shape=[])
  
  iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)
  
  x, y_, _ = iterator.get_next()
  train_op, loss, eval_op = model(x, y_)
  init = tf.initialize_all_variables()
  
  # summary
  logdir = './logs/m4d2a'
  def summary_op(datapart='train'):
  	tf.summary.scalar(datapart + '-loss', loss)
  	tf.summary.scalar(datapart + '-eval', eval_op)
  	return tf.summary.merge_all()	
  summary_op_train = summary_op()
  summary_op_test = summary_op('val')
  
  with tf.Session() as sess:
  	sess.run(init)
  	handle_train, handle_val, handle_test = sess.run(
  		[x.string_handle() for x in [iter_train, iter_val, iter_test]])
       _, cur_loss, cur_train_eval, summary = sess.run([train_op, loss, eval_op, summary_op_train], feed_dict={handle: handle_train, keep_prob: 0.5} )
       
      cur_val_loss, cur_val_eval, summary = sess.run([loss, eval_op, summary_op_test], 
  			feed_dict={handle: handle_val, keep_prob: 1.0})
  ```

  

  ###[Dataset输出数据不同迭代器介绍](https://blog.csdn.net/weixin_31767897/article/details/79365968)

  ​	

  ### 使用该方法读取变长数据

  [参考](https://blog.csdn.net/yeqiustu/article/details/79795639) 

  #### 和读取相同长度数据的不同之处

  1. 解析数据时用`tf.VarLenFeature(tf.datatype)`

  2. 使用该方法解析到的数据是一个稀疏tensor，所以要加一个`tf.sparse_tensor_to_dense`

  3. 使用padded_batch来指明各个数据成员要pad的形状，成员若是scalar，则用[ ]，若是list，则用[max_length]，若是array，则用[d1,...,dn]，假如各成员的顺序是scalar数据、list数据、array数据，则padded_shapes=([ ], [mx_length], [d1,...,dn])；

     

  ==例子==

  ```
  import tensorflow as tf
  
  train_f, val_f, test_f = ['mnist-%s.tfrecord'%i for i in ['train', 'val', 'test']]
  
  def parse_exmp(serial_exmp):
  	feats = tf.parse_single_example(serial_exmp, features {'feature':tf.VarLenFeature(tf.float32,
  		'label':tf.FixedLenFeature([10],tf.float32), 
  		'shape':tf.FixedLenFeature([], tf.int64)})
  	image = tf.sparse_tensor_to_dense(feats['feature']) #使用VarLenFeature读入的是一个 sparse_tensor，用该函数进行转换
  	
  	label = tf.reshape(feats['label'],[2,5])  #把label变成[2,5]，以说明array数据如何padding
  	
  	shape = tf.cast(feats['shape'], tf.int32)
  	return image, label, shape
  
  def get_dataset(fname):
  	dataset = tf.data.TFRecordDataset(fname)
  	return dataset.map(parse_exmp) # use padded_batch method if padding needed
  
  epochs = 16
  batch_size = 50  
  padded_shapes = ([784],[3,5],[]) #把image pad至784，把label pad至[3,5]，shape是一个scalar，不输入数字
  # training dataset
  dataset_train = get_dataset(train_f)
  dataset_train = dataset_train.repeat(epochs).shuffle(1000).padded_batch(batch_size, padded_shapes=padded_shapes)
  ```

  

  

  