import scipy.io
import scipy.signal
import sounddevice
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

# 1

t = np.linspace(0, 1, 2000)
Amplitudine = 3
Frecventa = 10
Faza_1 = 0
Faza_2 = 3*np.pi/2

x1 = Amplitudine * np.sin(2 * np.pi * Frecventa * t + Faza_1)
x2 = Amplitudine * np.cos(2 * np.pi * Frecventa * t + Faza_2)
fig, axs = plt.subplots(2)
fig.suptitle("Exercitiul 1")
axs[0].plot(t, x1)
axs[1].plot(t, x2)
fig.savefig("Exercitiu1.pdf")
plt.show()

