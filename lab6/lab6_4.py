import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

d = 13
x = np.random.rand(20)
t = np.linspace(0,1, 20)
f = np.sin(2*np.pi*x*t)
f_2 = np.concatenate((f[d:], f[:d]))
# Prima varianta are punctul de maxim la indexul 20-d
prima = np.argmax(np.real(np.fft.ifft(np.conjugate(np.fft.fft(f)) * np.fft.fft(f_2))))
# A doua varianta are un singur punct de 1 la indexul 20-d in rest este 0
adoua = np.argmax(np.real(np.fft.ifft(np.fft.fft(f_2)/np.fft.fft(f))))
print(20-prima)
fig, axs = plt.subplots(2)
axs[0].plot(t, f)
axs[1].plot(t, f_2)
plt.show()
