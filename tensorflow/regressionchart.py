import numpy as np
import matplotlib.pyplot as plt
import random
x = np.linspace(-1, 1, 100)
y = np.random.normal(0, 1, 100)
ynew = x + y
def LIP(x,y,xlab="X",ylab="Y"):
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    m,c,r,p,se1=stats.linregress(x,ynew)
    fig=plt.figure()
    cm1lab="$"+('y=%2.2fx+%2.2f, r^2=%1.2f'%(m,c,r**2))+"$";
    plt.plot(x,y,'^',mfc='none',mec='b',mew=1.2)
    plt.plot(x, m*x+c,'k--',linewidth=2,label=cm1lab)
    plt.ylabel(ylab,fontsize=16)
    plt.xlabel(xlab,fontsize=16)
    #plt.legend( loc='upper left')
    return(fig)
LIP(x,ynew)
plt.show()
