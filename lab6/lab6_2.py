import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

x = np.random.rand(100)

x_1 = scp.signal.fftconvolve(x,x)
x_2 = scp.signal.fftconvolve(x_1,x_1)
x_3 = scp.signal.fftconvolve(x_2,x_2)

fig, axs = plt.subplots(4, figsize=(9,14))
fig.suptitle("Exercitiul 2")
axs[0].plot(x)
axs[0].set_title("x")
axs[1].plot(x_1)
axs[1].set_title("x * x")
axs[2].plot(x_2)
axs[2].set_title("(x * x) * x")
axs[2].plot(x_2)
axs[3].set_title("((x * x) * x) * x")
axs[3].plot(x_3)
plt.savefig("6_2.pdf")