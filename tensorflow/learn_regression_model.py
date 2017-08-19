import tensorflow as tf
import numpy as np
# our model has only one feature column
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
# select the linear regression learning model
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# manually provide training and evaluation data sets
x_train = np.array([1,2,3,4])
y_train = np.array([-3,-5,-7,-9])
x_eval = np.array([5, 6, 7, 8])
y_eval = np.array([-11, -13, -15, -17])
# prepare the training input
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train, batch_size=4, num_epochs=1000)
# prepare the evaluation input											  
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)
# train the model with our training input
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate the loss with our evaluation input
results = estimator.evaluate(input_fn=eval_input_fn)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))
# predict a new value by using the trained model
x_predict = np.array([5, 6, 7, 8])
predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_predict}, num_epochs=1, shuffle=False)
predictions = estimator.predict(input_fn=predict_input_fn)
for i, p in enumerate(predictions):
  print("Prediction %s: %s" % (i + 1, p))

