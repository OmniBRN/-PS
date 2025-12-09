import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
np.random.seed(25)

x = np.linspace(-5, 5, 1000)
y_trend = x**2 / 20
y_sezon =  np.sin(2 * np.pi * 15 * x + np.pi/4) + 1/2 * np.sin(2 * np.pi * 6 * x)
y_noise = np.random.normal(size=1000)
y = y_trend + y_sezon + y_noise

# MSE pentru original[t+1] si predictie[t]
def MSE(original, predictie):  
    N = len(original)
    sum = 0
    for t in range(N-1):
        sum += (original[t+1]-predictie[t])**2
    return sum/N

def mediereExponentiala(y, alfa):
    N = len(y)
    s = np.zeros(N)
    s[0] = y[0]
    for t in range(1, N):
        s[t] = alfa * y[t] + (1-alfa) * s[t-1]
    return s

def mediereExponentialaDubla(y, alfa, beta, m=1):
    N = len(y)
    s = np.zeros(N)
    b = np.zeros(N)
    s[0] = y[0]
    b[0] = y[1] - y[0]
    sol = [0 for _ in range(N)]
    for t in range(1, N):
        s[t] = alfa*y[t] + (1-alfa)*(s[t-1] + b[t-1])
        b[t] = beta*(s[t] - s[t-1]) + (1-beta)*b[t-1]
    for t in range(m, N):
        sol[t] = s[t-m] + m*b[t-m]
    return sol
    
def mediereExponentialaTripla(y, alfa, beta, gamma,L=7, m=1):
    N = len(y)
    s = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    s[0] = y[0]
    b[0] = y[1] - y[0]
    c[0] = y[0]
    for t in range(1, N):
        if t >= L:
            s[t] = alfa*(y[t] - c[t-L]) + (1-alfa)*(s[t-1]+b[t-1])
            b[t] = beta*(s[t] - s[t-1]) + (1-beta)*b[t-1]
            c[t] = gamma*(y[t] - s[t] - b[t-1]) + (1-gamma)*c[t-L]
        else:
            s[t] = alfa*y[t] + (1-alfa)*(s[t-1]+b[t-1])
            b[t] = beta*(s[t] - s[t-1]) + (1-beta)*b[t-1]
            c[t] = gamma*(y[t] - s[t] - b[t-1])
    sol = [0 for _ in range(N)]
    for t in range(m, N):
        if t>=L:
            sol[t] = s[t-m] + m*b[t-m] + c[(t-m)-L+1+(m-1)%L]
        else:
            sol[t] = s[t-m] + m*b[t-m]
    return sol


    
    
# =========================================================================#
# Mediere Exponentiala

# Fixat initial cu 
# α = .2

# alfa = 0.2
# s = mediereExponentiala(y, alfa)
# eroare = MSE(y, s)
# fig, axs = plt.subplots(2)
# fig.suptitle(f"Mediere Exponentiala pentru α={0.2} cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[1].plot(x, s)
# plt.savefig("9_2_fixat.pdf")

# Algoritm plictisitor de cautare a lui alfa
# erori = []
# alfas = np.linspace(0, 1, 1000)
# smallest = 10e10
# best_alfa = -1
# for i, alfa in enumerate(alfas):
#     s = mediereExponentiala(y, alfa)
#     erori.append(MSE(y,s))
#     if smallest > erori[-1]:
#         smallest = erori[-1]
#         best_alfa = alfas[i]

# fig, axs = plt.subplots(figsize=(8,6))
# fig.suptitle("Grafic cu erorile pentru valori diferite de alfa")
# axs.plot(alfas, erori)
# plt.savefig("9_2_erori.pdf")

# best_alfa = 0.031031031031031032
# s = mediereExponentiala(y, best_alfa)
# eroare = MSE(y, s)
# fig, axs = plt.subplots(2, figsize=(8,6))
# fig.suptitle(f"Mediere Exponentiala pentru α={best_alfa} cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[0].set_title("Seria originala")
# axs[1].plot(x, s)
# axs[1].set_title("Predictia")
# plt.savefig("9_2_best_MD.pdf")

# =========================================================================#
# Mediere Exponentiala Dubla

# Fixat initial cu 
# α = 0.2, β = 0.3

# alfa = 0.2
# beta = 0.3
# s = mediereExponentialaDubla(y, alfa, beta)
# eroare = MSE(y, s)
# fig, axs = plt.subplots(2)
# fig.suptitle(f"Mediere Exponentiala pentru (α,β)=({0.2},{0.3}) cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[1].plot(x, s)
# plt.savefig("9_2_MED_fix.pdf")

# Algoritm plictisitor de cautare a lui alfa si beta
# erori = []
# alfas = np.linspace(0, 1, 100)
# betas = np.linspace(0, 1, 100)
# smallest = 10e10
# best_alfa = -1
# best_beta = -1
# for i, alfa in enumerate(alfas):
#     for j, beta in enumerate(betas):
#         s = mediereExponentialaDubla(y, alfa, beta)
#         eroare = MSE(y, s)
#         if smallest > eroare:
#             smallest = eroare 
#             best_alfa = alfas[i]
#             best_beta = betas[j]

# best_alfa, best_beta = 0.15151515151515152,0.15151515151515152
# s = mediereExponentialaDubla(y, best_alfa, best_beta)
# eroare = MSE(y, s)
# fig, axs = plt.subplots(2, figsize=(12,8))
# fig.suptitle(f"Mediere Exponentiala pentru (α,β) =({best_alfa},{best_beta}) cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[0].set_title("Seria originala")
# axs[1].plot(x, s)
# axs[1].set_title("Predictia")
# plt.savefig("9_2_MDE_best.pdf")
    

# =========================================================================#
# Mediere Exponentiala Tripla

# Fixat initial cu 
# α = 0.2, β = 0.3, γ = 0.4

# alfa = 0.2
# beta = 0.3
# gama = 0.4
# s = mediereExponentialaTripla(y, alfa, beta, gama)
# eroare = MSE(y, s)
# fig, axs = plt.subplots(2)
# fig.suptitle(f"Mediere Exponentiala pentru (α,β,γ)=({0.2},{0.3},{0.4}) cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[1].plot(x, s)
# plt.savefig("9_2_MET_fix.pdf")

# Algoritm plictisitor de cautare a lui alfa si beta si gama
# Si pe langa faptul ca e plictisitor ajungi sa folosesti putine valori pentru a nu incetinii
# programul prea tare

# erori = []
# alfas = np.linspace(0, 1, 22)
# betas = np.linspace(0, 1, 22)
# gamas = np.linspace(0, 1, 22)
# smallest = 10e10
# best_alfa = -1
# best_beta = -1
# best_gama = -1
# for i, alfa in enumerate(alfas):
#     for j, beta in enumerate(betas):
#         for k, gama in enumerate(gamas):
#             s = mediereExponentialaTripla(y, alfa, beta, gama)
#             eroare = MSE(y, s)
#             if smallest > eroare:
#                 smallest = eroare 
#                 best_alfa = alfas[i]
#                 best_beta = betas[j]
#                 best_gama = gamas[k]

# best_alfa = 0.14285714285714285
# best_beta = 0.14285714285714285
# best_gama = 0.047619047619047616
# s = mediereExponentialaTripla(y, best_alfa, best_beta, best_gama)
# eroare = MSE(y, s)
# fig, axs = plt.subplots(2, figsize=(12,8))
# fig.suptitle(f"Mediere Exponentiala pentru (α,β,γ)=({best_alfa},{best_beta},{best_gama}) cu MSE={eroare}")
# axs[0].plot(x, y)
# axs[0].set_title("Seria originala")
# axs[1].plot(x, s)
# axs[1].set_title("Predictia")
# plt.savefig("9_2_MET_best.pdf")