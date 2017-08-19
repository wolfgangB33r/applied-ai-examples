import tensorflow as tf

a = tf.placeholder(tf.float32, name = 'a')
b = tf.placeholder(tf.float32, name = 'b')
m = tf.multiply(a, b)
sess = tf.Session();
# create a file and serialize the session graph for TensorBoard
file_writer = tf.summary.FileWriter('./logs', sess.graph)
# run the model by filling its placeholder
print(sess.run(m, {a: 1.2, b: 3.5 }))
# fill placeholder by tensors
print(sess.run(m, {a: [5, 2], b: [2.5, 3.3] }))



