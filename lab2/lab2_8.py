import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

t = np.linspace((-1)*np.pi/2, np.pi/2, 10000)
x = np.sin(t)
eroarea = np.abs(t - x)
approx_pade = [(x - (7*x**3)/60)/(1+(x**2)/20) for x in t]

fig, axs = plt.subplots()
fig.suptitle("sin(α) = α")
axs.plot(t, x)
axs.plot(t, t)
plt.show()


fig, axs = plt.subplots()
fig.suptitle("Eroarea aproximarii sin(α) = α")
axs.plot(t, eroarea)
plt.show()

fig, axs = plt.subplots()
fig.suptitle("Pade approximation")
axs.plot(t, x)
axs.plot(t, approx_pade)
plt.show()