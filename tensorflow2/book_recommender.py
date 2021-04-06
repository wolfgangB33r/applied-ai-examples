from typing import Dict, Text

import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow_recommenders as tfrs

# Books are 1-8, for users, 1-6
# Rating from 1 to 5
ratings = pd.read_csv("ratings.csv")	

rtg = pd.read_csv("https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv")

MATRIX = np.zeros(shape=(len(ratings['book_id'].unique()),len(ratings['user_id'].unique())))

# Fill ratings matrix with the dedicated ratings
for index, row in ratings.iterrows():
	MATRIX[int(row['book_id'])-1][int(row['user_id'])-1] = row['rating']

RATINGS = tf.convert_to_tensor(MATRIX, dtype=tf.float32)
print(MATRIX)

# Prepare data
R = np.array(MATRIX)
N = len(MATRIX)
M = len(MATRIX[0])
K = 2 # number of hidden features
P = np.random.rand(N,K)
Q = np.random.rand(M,K)

# Create a tracing writer for TensorBoard
writer = tf.summary.create_file_writer('./logs')

class MatrixFactorizationModel:
    def __init__(self):
        self.P = tf.Variable(P, dtype=tf.float32, name='P')
        self.Q = tf.Variable(Q, dtype=tf.float32, name='Q')

    def __call__(self):
        return tf.matmul(self.P, self.Q, transpose_b=True)

# Mean squared loss function, to measure the quality of predictions 
@tf.function
def loss(r, r_pred):
    return tf.reduce_mean(tf.square(r - r_pred))

@tf.function
def train(model, ratings, lr=0.05):
    with tf.GradientTape() as t:
        current_loss = loss(ratings, model())
    
    dP, dQ = t.gradient(current_loss, [model.P, model.Q])
    model.P.assign_sub(lr * dP)
    model.Q.assign_sub(lr * dQ)

model = MatrixFactorizationModel()
# Start the TensorBoard tracing
tf.summary.trace_on(graph=True, profiler=True)
# Train the model
epochs = 50
for epoch in range(epochs):
    current_loss = loss(RATINGS, model())
    with writer.as_default(step=epoch):
        tf.summary.scalar('loss', current_loss)   
    train(model, RATINGS, lr=0.05)
    print(f"Epoch {epoch}: Loss: {current_loss.numpy()}")

# Print the model after training    
print(model())






