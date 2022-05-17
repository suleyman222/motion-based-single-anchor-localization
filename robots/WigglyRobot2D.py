import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # x+exp(−0.2x)sin(10x)
    x = np.linspace(0, 50, 100)
    plt.plot(np.sin(3.1*x)+x, x)
    plt.show()
