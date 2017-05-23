#
#
#
from numpy import exp, array, random, dot

import numpy as np
import matplotlib.pyplot as plt

# Calculates the Sigmoid function which represents our
# neuron's activation function.
def bin(x):
    if x < 0:
        return 0
    else:
        return 1

x = np.arange(-10., 10., 0.2)

a = []
for item in x:
    a.append(bin(item))

plt.xlim(x.min()*1.1, x.max()*1.1)
plt.ylim(-0.2, 1.1)
plt.plot(x, a)
plt.show()

