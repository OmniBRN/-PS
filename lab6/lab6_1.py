import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

Ts = [1, 1.5, 2, 4]
B = 2
t_sinc = np.linspace(-3, 3, 250)
sinc = np.pow(np.sinc(B*t_sinc),2)
t = [np.linspace(-3,0, int(np.ceil(3.5*x))) for x in Ts]

for i, l in enumerate(t):
    temp = l
    temp2 = np.flip(temp[:-1])
    temp2 = np.abs(temp2)
    temp = np.concatenate((temp, temp2))
    t[i] = temp

x = [np.pow(np.sinc(B*x), 2) for x in t]
# E gresit si nu-mi merge mintea sa-mi dau seama de ce
def get_matrix(t_array, Ts, N):
    res = []
    for n in range(N):
        temp = []
        for t in t_array:
            temp.append(np.sinc((t-n*Ts)/Ts))
        res.append(np.array(temp))
    return np.array(res)

M = get_matrix(t_sinc, 1, len(x[0]))

fig, axs = plt.subplots(2,2, figsize=(12,9))


axs[0][0].set_title("1Hz")
axs[0][0].set_xlabel("t[s]")
axs[0][0].set_ylabel("Amplitude")
axs[0][0].plot(t_sinc, sinc, color="black")
axs[0][0].stem(t[0], x[0])

axs[0][1].set_title("1.5Hz")
axs[0][1].set_xlabel("t[s]")
axs[0][1].set_ylabel("Amplitude")
axs[0][1].plot(t_sinc, sinc, color="black")
axs[0][1].stem(t[1], x[1])

axs[1][0].set_title("2Hz")
axs[1][0].set_xlabel("t[s]")
axs[1][0].set_ylabel("Amplitude")
axs[1][0].plot(t_sinc, sinc, color="black")
axs[1][0].stem(t[2], x[2])

axs[1][1].set_title("4Hz")
axs[1][1].set_xlabel("t[s]")
axs[1][1].set_ylabel("Amplitude")
axs[1][1].plot(t_sinc, sinc, color="black")
axs[1][1].stem(t[3], x[3])

plt.show()

