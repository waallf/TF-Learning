(项目源码链接)[https://github.com/yuxng/SubCNN_T  
因为他是对cuda8.0编译的，本机cuda9.0所以需要从新编译一遍  
1.将roi_pooling_op.cc ,roi_pooling_op_gpu.cu.cc,roi_pooling_op_gpu.h拷贝到`YOUR_TENSORFLOW_PATH/lib/python2.7/site-packages/tensorflow/core/user_ops`　　
  可以通过｀python -c 'import tensorflow as tf; print(tf.__file__)'｀查看tensorflow位置　　
２．`cd YOUR_TENSORFLOW_PATH/lib/python2.7/site-packages/tensorflow/core/user_ops/`　　
３．　运行以下命令
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

nvcc -std=c++11 -c -o roi_pooling_op_gpu.cu.o roi_pooling_op_gpu.cu.cc -I $TF_INC \
-I"/usr/local/lib/python3.5/dist-packages" -I"/usr/local" -D GOOGLE_CUDA=1 -x cu -Xcompiler \
-fPIC -DNDEBUG --expt-relaxed-constexpr


    
  
    
g++ -std=c++11 -shared roi_pooling_op.cc -o roi_pooling_op_gpu.so  roi_pooling_op_gpu.cu.o \
-I $TF_INC -fPIC -L /usr/local/cuda-9.0/lib64/ -lcudart -D_GLIBCXX_USE_CXX11_ABI=0 \
-L /usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework

```
##因为g++版本大于５.0，所以要加　-D_GLIBCXX_USE_CXX11_ABI=0

＃＃g++编译代码的时候需要使用tensorflow_framework.so动态库，所以在gcc+ 的参数中添加　　
-L /usr/local/lib/python3.5/dist-packages/tensorflow -ltensorflow_framework即可；

４．　最后测试是否成功
```
import tensorflow as tf
a = tf.load_op_library("/usr/local/lib/python3.5/dist-packages/tensorflow/user_ops/roi_pooling_op_gpu.so")
```
