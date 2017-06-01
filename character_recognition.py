#
# Example definition
# 5 by 8 grid of pixels for each character, means 40-element input vector of filled pixels
# and 26-element result vector indicating which of the 26 individual chars was detected.
# The neural network by definition has to offer 40 inputs and 26 outputs. Its a two layer 
# network with a layer of 10 neurons. Number of 10 was selected by some experience and a bit
# trial and error.     
# Font: Arial Bold Size 8
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
# Layer 1 (40 inputs with 10 neurons) initial random input weights
l1_input_weights = random.random((40,10))
# Layer 2 (1 neuron with 4 inputs) initial random input weights
l2_input_weights = random.random((10, 3))


def calcOutput(l1, l2, input):
    o_l1 = sig(dot(input, l1))
    o_l2 = sig(dot(o_l1, l2))
    return o_l1, o_l2


# Our training data sets
t_in = array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # character A
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # character B
			  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]]) # character C
			  
# Our training data sets outputs
t_out = array( [[1, 0, 0],
				[0, 1, 0],
				[0, 0, 1]]).T

for i in range(iterations):
    # calculate the output
    output_l1, output_l2 = calcOutput(l1_input_weights, l2_input_weights, t_in)
    
    # calculate the error for layer 2
    error_l2 = t_out - output_l2
    delta_l2 = error_l2 * sig_d(output_l2)
    
    # calculate the error for layer 1
    error_l1 = delta_l2.dot(l2_input_weights.T)
    delta_l1 = error_l1 * sig_d(output_l1)

    # adjust the weights according to the error and the derivate of the
    # activation function
    adjustment_l1 = dot(t_in.T, delta_l1)
    adjustment_l2 = dot(output_l1.T, delta_l2)
    # add the adjustment
    l1_input_weights += adjustment_l1
    l2_input_weights += adjustment_l2

# test run of our trained neural network with a new situation's input
print("Output in a new situation")
# classify a new input, that is a slightly modified character A
output_l1, output_l2 = calcOutput(l1_input_weights, l2_input_weights, array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]))

print(output_l2)


