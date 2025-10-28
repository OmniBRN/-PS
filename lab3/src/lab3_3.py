import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math

t = np.linspace(0, 1, 2000)
freq1 = 10
freq2 = 20
freq3 = 25
f1 = np.sin(2 * np.pi * freq1 * t)
f2 = np.sin(2 * np.pi * freq2 * t)
f3 = np.sin(2 * np.pi * freq3 * t)
f = f1 + f2 + f3

X = [0 for _ in range(30)]
for omega in range(30):
    sum = 0
    for i in range(len(f)-1):
        sum += f[i] * math.e**((1j * -2 * np.pi * i * omega)/len(f))
    X[omega] = sum


fig, axs = plt.subplots(1,2, figsize=(15,6)) 
fig.suptitle("Exercitiul 3")

axs[0].plot(t, f)
axs[0].set_title("Graficul functiei")
axs[0].set_xlabel("x")
axs[0].set_ylabel("f(x)")

axs[1].stem([i for i in range(30)], [np.abs(x) for x in X])
axs[1].set_title("Modulul Transformatei Fourier pentru omega in {0..29}")
axs[1].set_xlabel("omega")
axs[1].set_ylabel("|X[omega]|")
plt.savefig("3_3.pdf")