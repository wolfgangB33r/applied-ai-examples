import tensorflow as tf

# The function to be traced.
@tf.function
def sample_func(x, y):
  return tf.multiply(x, y)

writer = tf.summary.create_file_writer('./logs')

# Sample data, implicitly typed as float32
x = tf.constant(4.0) 
y = tf.constant(6.0) 

# Tracing starts with calling tf.summary.trace_on() 
# and ends with a call to tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = sample_func(x, y)
# Print the result
print(z) 
print(float(z))
with writer.as_default():
  tf.summary.trace_export(name="sample_func_trace", step=0, profiler_outdir="./logs")