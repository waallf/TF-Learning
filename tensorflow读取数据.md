

[参考](http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/reading_data.html)

## TF的数据读取方式：

1. 供给数据(Feeding)  :  在TensorFlow程序运行的每一步， 让Python代码来供给数据 
2. 从文件读取数据： 在TensorFlow图的起始， 让一个输入管线从文件中读取数据 
3. 预加载数据 ：在TensorFlow图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况)。 



## 数据供给

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

# tf.data

