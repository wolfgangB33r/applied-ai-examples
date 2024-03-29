import tensorflow as tf
import matplotlib.pyplot as plt

writer = tf.summary.create_file_writer('./logs')
tf.summary.trace_on(graph=True, profiler=True)

class LinearRegressionModel:
    def __init__(self):
        self.k = tf.Variable(16.0)
        self.d = tf.Variable(10.0)

    def __call__(self, x):
        return self.k * x + self.d

# Mean squared loss function, to measure the quality of predictions 
@tf.function
def loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

@tf.function
def train(model, X, y, lr=0.05):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(X))
    dk, dd = t.gradient(current_loss, [model.k, model.d])
    model.k.assign_sub(lr * dk)
    model.d.assign_sub(lr * dd)

model = LinearRegressionModel()
print(model(1))

# Let's generate some example data
true_k = 1.3 # slope
true_d = 0.3 # intercept

NUM_EXAMPLES = 100
X = tf.random.normal(shape=(NUM_EXAMPLES,))
noise = tf.random.normal(shape=(NUM_EXAMPLES,))
y = X * true_k + true_d + noise

# Plot before learning the model
import matplotlib.pyplot as plt
plt.scatter(X, y, label="true")
plt.scatter(X, model(X), label="predicted")
plt.legend()
plt.show()

# Training the model
epochs = 50
for epoch in range(epochs):
    with writer.as_default(step=epoch):
        current_loss = loss(y, model(X))
        tf.summary.scalar('loss', current_loss)   
        train(model, X, y, lr=0.1)
        print(f"Epoch {epoch}: Loss: {current_loss.numpy()}")
    
# Plot after training the model
plt.scatter(X, y, label="true")
plt.scatter(X, model(X), label="predicted")
plt.legend()
plt.show()


  