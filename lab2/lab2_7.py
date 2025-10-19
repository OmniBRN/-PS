import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

t = np.linspace(0,1, 1000)

x = np.sin(2 * np.pi * t * 250)
x_decimat = x[::4]
t_decimat = t[::4]
x_decimat_2 = x[1::4]
t_decimat_2 = t[1::4]

fig, axs = plt.subplots()
fig.suptitle("Graficul initial - 250Hz")
axs.plot(t,x)
plt.savefig("2_7_graficul_initial.pdf")

fig, axs = plt.subplots()
fig.suptitle("Graficul decimat")
axs.plot(t_decimat,x_decimat)
plt.savefig("2_7_graficul_decimat.pdf")

fig, axs = plt.subplots()
fig.suptitle("Graficul decimat cu offset 1")
axs.plot(t_decimat_2,x_decimat_2)
plt.savefig("2_7_graficul_decimat_2.pdf")

# Atunci cand faci offset la decimare rezultatul este acelasi grafic doar 
# ca in alta faza si ambele sunt sinusoidale cu frecventa mai mica 
# ca graficul initial