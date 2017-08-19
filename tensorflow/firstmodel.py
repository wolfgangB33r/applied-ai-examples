import tensorflow as tf

a = tf.constant(4.0) # both nodes are implicitly typed as float32
b = tf.constant(6.0)
m = tf.multiply(a, b)

sess = tf.Session();
print(sess.run(m))
