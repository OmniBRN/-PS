import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
np.random.seed(25)

x = np.linspace(-5, 5, 1000)
y_trend = x**2 / 20
y_sezon =  np.sin(2 * np.pi * 15 * x + np.pi/4) + 1/2 * np.sin(2 * np.pi * 6 * x)
y_noise = np.random.normal(size=1000)
y = y_trend + y_sezon + y_noise

def MSE(x, y):
    return np.mean(np.square(np.subtract(x,y)))

def mediereExponentiala(y, alfa):
    N = len(x)
    s = np.zeros(N)
    s[0] = y[0]
    for t in range(1, N):
        s[t] = alfa * y[t] + (1-alfa) * y[t-1]
    return s

# Fixat initial cu 
# α = .2

# alfa = 0.2
# s = mediereExponentiala(y, alfa)
# eroare = MSE(x, s)
# fig, axs = plt.subplots(2)
# fig.suptitle(f"Mediere Exponentiala pentru α={0.2} cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[1].plot(x, s)
# plt.savefig("9_2_fixat.pdf")

# erori = []
# alfas = np.linspace(0, 1, 500)
# smallest = 10e10
# best_alfa = -1
# for i, alfa in enumerate(alfas):
#     s = mediereExponentiala(y, alfa)
#     erori.append(MSE(x,s))
#     if smallest > erori[-1]:
#         smallest = erori[-1]
#         best_alfa = alfas[i]

# fig, axs = plt.subplots(figsize=(8,6))
# fig.suptitle("Grafic cu erorile pentru valori diferite de alfa")
# axs.plot(alfas, erori)
# plt.savefig("9_2_erori.pdf")

best_alfa = 0.5070140280561122
s = mediereExponentiala(y, 0.2)
eroare = MSE(x, s)
fig, axs = plt.subplots(2)
fig.suptitle(f"Mediere Exponentiala pentru α={0.2} cu MSE={eroare}")
axs[0].plot(x, y)
axs[1].plot(x, s)
plt.savefig("9_2_fixat.pdf")
