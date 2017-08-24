# books are 1-10000, for users, 1-53424
# rating from 1 to 5
import numpy
import tensorflow as tf
	
R_BOOKS = [ # columns = user, rows = books
     [4,0,0,0,2,0],
     [0,0,0,3,5,0],
     [0,1,0,5,4,0],
     [1,4,0,0,0,5],
     [0,5,0,0,0,0],
    ]
	
# prepare data
R = numpy.array(R_BOOKS)
N = len(R_BOOKS)
M = len(R_BOOKS[0])
K = 2 # number of hidden features
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

# input placeholders  
ratings = tf.placeholder(tf.float32, name = 'ratings')

# model variables
tP = tf.Variable(P, dtype=tf.float32, name='P')
tQ = tf.Variable(Q, dtype=tf.float32, name='Q')
  
# build model
pDotq = tf.tensordot(tP, tQ, [[1], [1]]) 
  
squared_deltas = tf.square(pDotq - ratings)
loss = tf.reduce_sum(squared_deltas)
tf.summary.scalar('loss', loss)
tf.summary.scalar('sumP', tf.reduce_sum(tP))
tf.summary.scalar('sumQ', tf.reduce_sum(tQ))
   
# create an optimizer towards minimizing the loss value
optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)
	
sess = tf.Session();
init = tf.global_variables_initializer()
sess.run(init)

# write summaries to TensorBoard log directory
summary_op = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('./logs', sess.graph)
	
# now run the training loop to reduce the loss
for i in range(5000):
    # also execute a summary operation together with the train node for TensorBoard
    _, summary = sess.run([train, summary_op], {ratings: R_BOOKS})
    file_writer.add_summary(summary, i)
  
print(sess.run(pDotq))




