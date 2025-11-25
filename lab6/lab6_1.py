import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

Ts = [1, 1.5, 2, 4]
B = 1
t_sinc = np.linspace(-3, 3, 250)
sinc = np.pow(np.sinc(B*t_sinc),2)
t = [np.linspace(-3,0, int(np.ceil(3.5*x))) for x in Ts]
for i, l in enumerate(t):
    temp = l
    temp2 = np.flip(temp[:-1])
    temp2 = np.abs(temp2)
    temp = np.concatenate((temp, temp2))
    t[i] = temp
t_2 = [np.arange(-3, 3 + 1/Ts_i, 1/Ts_i) for Ts_i in Ts]

x = [np.pow(np.sinc(B*t_array), 2) for t_array in t]
x_2 = [np.pow(np.sinc(B*t_array), 2) for t_array in t_2]

def get_sinc_reconstruit(t_dens, t_array, x, Ts):
    T = 1 / Ts
    return np.sum(
        x[:, None] * np.sinc((t_dens[None, :] - t_array[:, None]) / T),
        axis=0
    )

x_caciula = [get_sinc_reconstruit(t_sinc, t_2[y], x_2[y], Ts[y]) for y in range(4)]


fig, axs = plt.subplots(2,2, figsize=(12,9))

fig.suptitle("Exercitiul 1")
axs[0][0].set_title("Ts = 1Hz")
axs[0][0].set_xlabel("t[s]")
axs[0][0].set_ylabel("Amplitude")
axs[0][0].plot(t_sinc, sinc, color="black")
axs[0][0].plot(t_sinc, x_caciula[0], color="green")
axs[0][0].stem(t[0], x[0])

axs[0][1].set_title("Ts = 1.5Hz")
axs[0][1].set_xlabel("t[s]")
axs[0][1].set_ylabel("Amplitude")
axs[0][1].plot(t_sinc, sinc, color="black")
axs[0][1].plot(t_sinc, x_caciula[1], color="green")
axs[0][1].stem(t[1], x[1])

axs[1][0].set_title("Ts = 2Hz")
axs[1][0].set_xlabel("t[s]")
axs[1][0].set_ylabel("Amplitude")
axs[1][0].plot(t_sinc, sinc, color="black")
axs[1][0].plot(t_sinc, x_caciula[2], color="green")
axs[1][0].stem(t[2], x[2])

axs[1][1].set_title("Ts = 4Hz")
axs[1][1].set_xlabel("t[s]")
axs[1][1].set_ylabel("Amplitude")
axs[1][1].plot(t_sinc, sinc, color="black")
axs[1][1].plot(t_sinc, x_caciula[3], color="green")
axs[1][1].stem(t[3], x[3])

plt.savefig("6_1.pdf")

