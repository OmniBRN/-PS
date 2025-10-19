import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

fs = 2000
t = np.linspace(0, 1, fs)

x_a = np.sin(2*np.pi * fs/2 * t)
x_b = np.sin(2*np.pi * fs/4 * t)
x_c = np.sin(0 * t)

fig, axs = plt.subplots(3)
fig.suptitle("Exercitiul 6 - Frecvente fundamentale diferite")
axs[0].plot(t, x_a)
axs[1].plot(t, x_b)
axs[2].plot(t, x_c)
plt.savefig("2_6.pdf")
plt.show()