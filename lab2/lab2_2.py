import scipy.io
import scipy.signal
import sounddevice
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

t = np.linspace(0,1, 2000)
frecventa = 8
faze = np.pi * np.array([3/4, 1, 1/2, 1/8])
grafice = [np.sin(2 * np.pi * frecventa * t + x) for x in faze]
fig, axs = plt.subplots()
fig.suptitle("Exercitiul 2 - Cele Patru Faze")
for i in range(len(faze)):
    axs.plot(t, grafice[i])
axs.set_xlabel("Time")
axs.set_ylabel("Amplitude")
plt.savefig("2_2_Cele_Patru_Faze.pdf")

# Alegem al doilea

z = np.random.normal(loc = 0.0, scale=1.0, size = 2000)
norma_x = np.power(np.linalg.norm(grafice[1],2),2)
norma_z = np.power(np.linalg.norm(z, 2),2)

SNR = [0.1, 1, 10, 100]
gamma = [np.sqrt(t) for t in [norma_x/(x * norma_z) for x in SNR]]

grafice_finale = [grafice[1] + gamma[i] * z for i in range(len(SNR))]
fig, axs = plt.subplots(4, figsize=(16,12))
fig.suptitle("Exercitiul 2 - Graficul 2 cu Noise")
for i in range(len(SNR)):
    axs[i].plot(t, grafice_finale[i])
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Amplitude")

plt.savefig("2_2_Noise_Graph.pdf")

