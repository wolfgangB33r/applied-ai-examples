#
# Just to plot a function with multiple parameter
#
from numpy import exp, array, random, dot

import numpy as np
import matplotlib.pyplot as plt

# Calculates a function based on three parameter.
def f(a,b):
    return (sin(a) + cos(b))/c;

x = np.arange(-10., 10., 0.2)

a = []
for item in x:
    a.append(sig(item))

plt.xlim(x.min()*1.1, x.max()*1.1)
plt.ylim(-0.2, 1.1)

plt.plot(x, a)
plt.show()

