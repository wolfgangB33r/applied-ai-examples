import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

true_k = 1.3 # slope
true_d = 0.3 # intercept

NUM_EXAMPLES = 100
X = tf.random.normal(shape=(NUM_EXAMPLES,))
noise = tf.random.normal(shape=(NUM_EXAMPLES,))
y = X * true_k + true_d + noise

train_x = np.array(X)

# Register TensorBoard tracing callback
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')

# We use a single-variable linear regression, to predict the y values from a given X values.
reg_normalizer = preprocessing.Normalization(input_shape=[1,])
reg_normalizer.adapt(train_x)

regression_model = tf.keras.Sequential([
    reg_normalizer,
    layers.Dense(units=1)
])

regression_model.summary()

print(regression_model.predict(X[:10]))

regression_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = regression_model.fit(
    X, y,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2,
    callbacks=[tensorboard_callback])

# Plot before learning the model
import matplotlib.pyplot as plt
plt.scatter(X, y, label="true")
plt.scatter(X, regression_model.predict(X), label="predicted")
plt.legend()
plt.show()
