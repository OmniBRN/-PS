import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

t = np.linspace((-1)*np.pi/2, np.pi/2, 10000)
x = np.sin(t)
eroarea = np.abs(t - x)
approx_pade = [(x - (7*x**3)/60)/(1+(x**2)/20) for x in t]

fig, axs = plt.subplots()
fig.suptitle("Exercitiul 8 - Aproximarea sin(α) = α")
axs.plot(t, x)
axs.plot(t, t)
axs.set_xlabel("Time")
axs.set_ylabel("Amplitude")
plt.savefig("2_8_aproximarea_1.pdf")

fig, axs = plt.subplots()
fig.suptitle("Exercitiul 8 - Eroarea aproximarii sin(α) = α (|sin(α) - α|)")
axs.plot(t, eroarea)
axs.set_xlabel("Time")
axs.set_ylabel("Error")
plt.savefig("2_8_Eroarea_Aproximarii_1.pdf")

fig, axs = plt.subplots()
fig.suptitle("Exercitiul 8 - Pade approximation")
axs.plot(t, x)
axs.plot(t, approx_pade)
axs.set_xlabel("Time")
axs.set_ylabel("Amplitude")
plt.savefig("2_8_Aproximarea_Pade.pdf")

fig, axs = plt.subplots()
fig.suptitle("Exercitiul 8 - sin(α) = α (Scara Logaritmica)")
axs.plot(t, x)
axs.plot(t, t)
axs.set_xlabel("Time")
axs.set_ylabel("Amplitude")
plt.yscale("log")
plt.savefig("2_8_aproximarea_1_logaritmica.pdf")

fig, axs = plt.subplots()
fig.suptitle("Exercitiul 8 - Eroarea aproximarii sin(α) = α (|sin(α) - α|) (Scara Logaritmica)")
axs.plot(t, eroarea)
axs.set_xlabel("Time")
axs.set_ylabel("Error")
plt.yscale("log")
plt.savefig("2_8_Eroarea_Aproximarii_1_logaritmica.pdf")

fig, axs = plt.subplots()
fig.suptitle("Exercitiul 8 - Pade approximation (Scara Logaritmica)")
axs.plot(t, x)
axs.plot(t, approx_pade)
axs.set_xlabel("Time")
axs.set_ylabel("Amplitude")
plt.yscale("log")
plt.savefig("2_8_Aproximarea_Pade_logaritmica.pdf")