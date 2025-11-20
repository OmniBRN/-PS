import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

t = np.linspace(0,1,1000)
f = 100
A = 1
f = A*np.sin(2*np.pi*f*t)
x = np.linspace(0, 200, 200)

fereastra_drept = np.concatenate((np.zeros(400),np.ones(200), np.zeros(400)))
fereastra_hanning = np.concatenate((np.zeros(400),0.5 * (1 - np.cos(2*np.pi*x/200)),np.zeros(400)))

fig, axs = plt.subplots(3)
axs[0].plot(t,f)
axs[1].plot(t,f*fereastra_drept)
axs[2].plot(t,f*fereastra_hanning)
plt.show()