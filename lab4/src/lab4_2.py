import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math

# fs = 20 a unui semnal de f = 4 -> alierea se intampla la frecvente de 4+(20-1)n 
t1 = np.linspace(0,1,20)
t2 = np.linspace(0,1,2000)
f1s = np.sin(2*np.pi*4*t1)
f1 = np.sin(2*np.pi*4*t2)
f2 = np.sin(2*np.pi*23*t2)

fig, axs = plt.subplots(figsize = (8,6))
fig.suptitle("Exercitiul 2")
axs.plot(t2, f1,label="3Hz")
axs.plot(t2, f2,label="23Hz")
axs.stem(t1, f1s)
axs.set_xlabel("Timp(s)")
axs.set_ylabel("Amplitudine")
axs.legend()
plt.savefig("4_2.pdf")




