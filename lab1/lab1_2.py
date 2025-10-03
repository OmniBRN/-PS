import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('qtAgg')

# # 1. a)
# x = np.arange(0, 0.03+0.0005, 0.0005)

# # 1. b)

# y_x = np.cos(520 * np.pi * x + np.pi/3)
# y_y = np.cos(280 * np.pi * x - np.pi/3)
# y_z = np.cos(120 * np.pi * x + np.pi/3)

# fig, axs = plt.subplots(3)
# fig.suptitle('Continuu')
# axs[0].plot(x,y_x)
# axs[1].plot(x,y_y)
# axs[2].plot(x,y_z)
# plt.show()

# # 1. c)
# Ts = 1/200
# t_e = np.arange(0, 0.03+Ts, Ts)
# x_e = np.cos(520*np.pi*t_e + np.pi/3)
# y_e = np.cos(280*np.pi*t_e - np.pi/3)
# z_e = np.cos(120*np.pi*t_e + np.pi/3)

# fig2, axs2 = plt.subplots(3)
# fig2.suptitle('Esantionate')
# axs2[0].stem(t_e, x_e)
# axs2[1].stem(t_e, y_e)
# axs2[2].stem(t_e, z_e)
# plt.show()

# 2.a)
Ts1 = 1/400
t1 = np.linspace(0, 1, 1600)
x = np.sin(2*np.pi*1/Ts1*t1)

# 2.b)
Ts2 = 1/800
t2 = np.linspace(0, 3)
y = np.sin(2*np.pi*1/Ts2*t2)

# 2.c)
Ts3 = 1/240
t3 = np.linspace(0, 3, 720)
z = np.mod(t3, 1)

# 2.d)
Ts4 = 1/300
t4 = np.linspace(0,3, 900)
t = np.sign(t4)

fig, axs = plt.subplots(4)
fig.suptitle('test')
axs[0].plot(t1,x)
axs[1].plot(t2,y)
axs[2].plot(t3,z)
axs[3].plot(t4,t)

plt.show()

