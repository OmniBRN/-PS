import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math

t1 = np.linspace(0,1,8)
t1_2 = np.linspace(0,1,25)
t2 = np.linspace(0,1,2000)
f1s = np.sin(2*np.pi*4*t1)
f1s2 = np.sin(2*np.pi*4*t1_2)
f1 = np.sin(2*np.pi*4*t2)
f2 = np.sin(2*np.pi*11*t2)

fig, axs = plt.subplots(figsize=(8,6))
fig.suptitle("Exercitiul 2")
axs.plot(t2, f1,label="4Hz", color="#71C460")
axs.plot(t2, f2,label="11Hz", color="#8B0000")
axs.stem(t1, f1s, label="fs = 8")
axs.stem(t1_2, f1s2, linefmt="C2-", label="fs = 25")
axs.set_xlabel("Timp(s)")
axs.set_ylabel("Amplitudine")
axs.legend()
plt.savefig("4_3.pdf")