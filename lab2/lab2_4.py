import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

# Graficul 1 - f(x) = 3 * sin(2pi* 21 * x + 4pi/3)
t = np.linspace(0,1, 20000)
amplitudine_sinus = 3
faza_sinus = 4*np.pi/3
frecventa_sinus = 21
x_sinus = amplitudine_sinus * np.sin(2*np.pi * frecventa_sinus * t + faza_sinus)

# Graficul 2 - f(x) = 20 * mod(x, 1/20)
y_sawtooth = 20 * np.mod(t, 1/20)

z_suma = x_sinus + y_sawtooth

fig, axs = plt.subplots(3, figsize=(12,9))
fig.suptitle("Exercitiul 4 - Sinus, Sawtooth si Suma lor")
axs[0].plot(t, x_sinus)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Amplitude")
axs[1].plot(t, y_sawtooth)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Amplitude")
axs[2].plot(t, z_suma)
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Amplitude")
plt.savefig("2_4.pdf")
plt.show()

