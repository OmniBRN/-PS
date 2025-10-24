import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math

t = np.linspace(0, 1, 5000)
frecventa = 6

x = np.sin(2 * np.pi * t * frecventa)
y = x * math.e**(1j * -2 *np.pi * t)
y_real = [t.real for t in y]
y_imag = [t.imag for t in y]

fig, axs = plt.subplots(figsize=(6,6))
fig.suptitle("Exercitiul 2 - y[n]")
axs.plot(y_real, y_imag)
plt.savefig("3_2_y[n].pdf")

fig, axs = plt.subplots(2,2, figsize=(8,8))


for i, omega in enumerate([2, 4, 6, 8]):
    z = x * math.e**(1j * -2 * np.pi * t * omega)
    axs[int(i/2)][i%2].plot([t.real for t in z], [t.imag for t in z])
    axs[int(i/2)][i%2].set_title(f"omega = {omega}")
plt.savefig("3_2_z[n].pdf")

def distanta(x, y):
    return np.sqrt(x**2 + y**2)

def getColor(distance):
    norm = distance/np.sqrt(2)
    return (norm, 0, 1-(norm/2 + 0.5))

fig, axs = plt.subplots(figsize=(6,6))
fig.suptitle("Exercitiul 2 - y[n]")
colors = [getColor(distanta(t1.real, t1.imag)) for t1 in y]
axs.scatter(y_real, y_imag, c=colors)
plt.savefig("3_2_y[n]_colored.pdf")

fig, axs = plt.subplots(2,2, figsize=(8,8))
for i, omega in enumerate([2, 4, 6, 8]):
    z = x * math.e**(1j * -2 * np.pi * t * omega)
    z_real = [t.real for t in z]
    z_imag = [t.imag for t in z]
    colors2 = [getColor(distanta(t1.real, t1.imag)) for t1 in z]
    axs[int(i/2)][i%2].scatter(z_real,z_imag, c=colors2)
    axs[int(i/2)][i%2].set_title(f"omega = {omega}")
plt.savefig("3_2_z[n]_colored.pdf")



