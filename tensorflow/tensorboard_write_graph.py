import tensorflow as tf

a = tf.constant(4.0) # both nodes are implicitly typed as float32
b = tf.constant(6.0)
m = tf.multiply(a, b)
sess = tf.Session();
# create a file and serialize the session graph for TensorBoard
file_writer = tf.summary.FileWriter('./logs', sess.graph)
print(sess.run(m))
