import tensorflow as tf
data = [1,2,3,4,5,6,7,8,9,0,10,11,13]
pl = tf.placeholder(dtype=tf.int32,shape=[None])
s = tf.train.shuffle_batch()
tf.summary.his