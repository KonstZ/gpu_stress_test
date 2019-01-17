
#To use TensorCores a special version of tensorflow is required
#It can be built from source or installed from Nvidia's docker container https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/index.html
import tensorflow as tf

DTYPE_1 = 'float32' 
#uncomment this to make use of TensorCores
#DTYPE_1 = 'float16'  

DTYPE_2 = 'float32' 
SIZE = 1024 * 16
SIZE = (SIZE, SIZE)

def test_gpu(count):
	with tf.Session() as sess:
		y = []
		for i in range(count):
			with tf.device('/gpu:' + str(i)):
				x = tf.random.normal(SIZE, dtype=DTYPE_1)
				x = tf.matmul(x, x)
				x = tf.cast(x, dtype=DTYPE_2)
				x = tf.matmul(x, x)
				y.append(tf.reduce_mean(x))
		
		while(True):
			sess.run(y)

import sys
test_gpu(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

