import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qtAgg')

# 1. a)
x = np.linspace(0, 0.03, int(0.03/0.0005))

# 1. b)

y_x = np.cos(520 * np.pi * x + np.pi/3)
y_y = np.cos(280 * np.pi * x - np.pi/3)
y_z = np.cos(120 * np.pi * x + np.pi/3)

fig, axs = plt.subplots(3)
fig.suptitle('Continuu')
axs[0].plot(x,y_x)
axs[1].plot(x,y_y)
axs[2].plot(x,y_z)
plt.xlabel("Timp")
plt.ylabel("Semnal")
fig.savefig("1_Continuu.pdf")
plt.show()

# 1. c)
Ts = 1/200
t_e = np.linspace(0, 0.03, 6)
x_e = np.cos(520*np.pi*t_e + np.pi/3)
y_e = np.cos(280*np.pi*t_e - np.pi/3)
z_e = np.cos(120*np.pi*t_e + np.pi/3)

fig2, axs2 = plt.subplots(3)
fig2.suptitle('Esantionate')
axs2[0].stem(t_e, x_e)
axs2[1].stem(t_e, y_e)
axs2[2].stem(t_e, z_e)
plt.xlabel("Timp")
plt.ylabel("Semnal")
fig2.savefig("1_Esantionate.pdf")
plt.show()
