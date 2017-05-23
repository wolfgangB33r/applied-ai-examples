#
#
#
from numpy import exp, array, random, dot

# Calculates the Sigmoid function which represents our
# neuron's activation function.
def sig(x):
    return 1 / (1 + exp(-x))
    
# The derivative of the Sigmoid function.
# This is the gradient of the Sigmoid curve.
# It indicates how confident we are about the existing weight.
def sig_d(x):
    return x * (1 - x)


# initilize the seed with a fixed value to generate the same
# results in consecutive runs
random.seed(1)
# How many training iterations do we perform?
iterations = 10000
# Initial random input weights
input_weights = random.random((3,1))
# Our training data sets
t_in = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
# Our training data sets outputs
t_out = array([[0, 1, 1, 0]]).T

for i in xrange(iterations):
    # calculate the output
    output = sig(dot(t_in, input_weights))
    # calculate the error
    error = t_out - output
    # adjust the weights according to the error and the derivate of the
    # activation function
    adjustment = dot(t_in.T, error * sig_d(output))
    # add the adjustment
    input_weights += adjustment

# test run of our trained neuron with a new situations input
print(input_weights)
print("Output for a new situation")
print(sig(dot([1, 0, 0], input_weights)))


