# 将数据转为tfrecord形式  
## tfrecord介绍：  
tfrecord包含了tf.train.Example协议缓冲区，先获取数据，然后将数据填  
入到Example协议缓冲区，将协议缓冲区序列化为一个字符串，然后通过tf.python_io.TFRecordWriter写入TFRecoed  


1. tf转换
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

# 读取tfrecord  
从TFRecord文件中读取数据，需要首先使用tf.train.string_input_producer生成一个解析队列  
然后使用tf.TFRecordReader读取生成的解析队列  
在使用tf.paese_single_example()解析器来解析队列  
```
tfrecords_filename = "train.tfrecords" 
file_queue = tf.train.string_input_producer([tfrecords_filenmae],)#生成解析队列
reader = tf.TFRecordReader()
_,serialiezd_example = reader.read(file_queue)
feature = tf.parse_single_example(serialiezd_example,
                                  features={
                                  'label':tf.FixedLenFeature([],tf.int64),
                                  "ima_raw":tf.FixedLenFeature([],tf.string),
  
  })
  
img = feature['label']
label = feature['ima_raw']

                                    
```  
*注意一下两句*  
```
with tf.Session() as sess:
  coord=tf.train.Coordinator()
  threads= tf.train.start_queue_runners(coord=coord)
  for i in range(20):
    example, l = sess.run([image,label])#必须开启了队列才可以读取
  coord.request_stop()
  coord.join(threads)
```
