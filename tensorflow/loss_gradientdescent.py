import tensorflow as tf

k = tf.Variable(0.5, dtype=tf.float32, name='k')  # slope variable, initialized with 0.5
tf.summary.scalar('slope', k)
d = tf.Variable(-0.12, dtype=tf.float32, name='d') # y-intersect, initialized with -0.12
tf.summary.scalar('y-intersect', d)
x = tf.placeholder(tf.float32, name = 'x')
linear_model = k * x + d

sess = tf.Session();
init = tf.global_variables_initializer()
sess.run(init)
# calculate the loss between our model and given values for y
y = tf.placeholder(tf.float32, name = 'y')
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
tf.summary.scalar('loss', loss)

# create an optimizer towards minimizing the loss value
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# write summaries to TensorBoard log directory
summary_op = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('./logs', sess.graph)

# now run the training loop to reduce the loss
for i in range(1000):
    # also execute a summary operation together with the train node for TensorBoard
    _, summary = sess.run([train, summary_op], {x:[1,2,3,4], y:[-3,-5,-7,-9]})
    file_writer.add_summary(summary, i)


print('Parameters after training: ')
print(sess.run([k, d]))
print(sess.run(loss, {x:[1,2,3,4], y:[-3,-5,-7,-9]}))

