import matplotlib.pyplot as plt
import numpy as np

def step(x):
    y = []
    for i in x:
        y.append(1 * (i > 0))
    return y


def sigmoid(x):
    y = []
    for i in x:
        y.append(1 / (1 + np.exp(-i)))
    return y


def rect(x):
    y = []
    for i in x:
        y.append(np.maximum(0, i))
    return y


x = np.linspace(-5, 5, 100)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

fig.suptitle('Activation functions')
ax1.set_ylabel('output')
ax2.set_xlabel('input')
ax1.set_title('Step')
ax1.plot(x, step(x))
ax2.set_title('Sigmoid')
ax2.plot(x, sigmoid(x))
ax3.set_title('Rectifier')
ax3.plot(x, rect(x))

plt.show()
