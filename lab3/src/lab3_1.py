import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math

def getFourierMatrix(order=8):
    result = []
    for i in range(0, order):
        t = np.array([])
        for j in range(0, order):
            t = np.append(t, math.e**(-2j*np.pi*i*j/order))
        print(t)
        result.append(t.copy())
    return np.array(result)


order = 2 
matrix = getFourierMatrix(order)

fig, axs = plt.subplots(order)
fig.suptitle("Exercitiul 1 - Reprezentarea Transformatei Fourier")
for i,x in enumerate(matrix):
    real_x = np.array([y.real for y in x])
    imaginary_x = np.array([y.imag for y in x])
    axs[i].plot(real_x)
    axs[i].plot(imaginary_x)

plt.savefig("3_1.pdf")
plt.show()

matrix_2 = 1/order * np.conjugate(np.transpose(matrix))
result = matrix_2@matrix
print(np.allclose(result, np.eye(order)))





