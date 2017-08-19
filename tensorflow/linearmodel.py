import tensorflow as tf

k = tf.Variable(.5, dtype=tf.float32, name='k')  # slope variable, initialized with 0.5
d = tf.Variable(-.12, dtype=tf.float32, name='d') # y-intersect, initalized with -0.12
x = tf.placeholder(tf.float32, name = 'x')
linear_model = k * x + d


sess = tf.Session();
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x : [1, 2, 3, 4]}))
# create a file and serialize the session graph for TensorBoard
file_writer = tf.summary.FileWriter('./logs', sess.graph)
