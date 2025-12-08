import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
np.random.seed(25)

# a)

x = np.linspace(-5, 5, 1000)
y_trend = x**2 / 20
y_sezon =  np.sin(2 * np.pi * 15 * x + np.pi/4) + 1/2 * np.sin(2 * np.pi * 6 * x)
y_noise = np.random.normal(size=1000)
y = y_trend + y_sezon + y_noise

fig, axs = plt.subplots(2,2, figsize=(20, 10))
fig.suptitle("Seria de Timp Aleatoare")
axs[0][0].plot(x, y, label="Seria de timp")
axs[0][0].set_xlabel("Timp")
axs[0][0].set_ylabel("Amplitudine")
axs[0][0].set_title("Seria de timp")
axs[0][1].plot(x, y_trend, label="Trend")
axs[0][1].set_xlabel("Timp")
axs[0][1].set_ylabel("Amplitudine")
axs[0][1].set_title("Trend")
axs[1][0].plot(x, y_sezon, label="Sezon")
axs[1][0].set_title("Sezon")
axs[1][0].set_xlabel("Timp")
axs[1][0].set_ylabel("Amplitudine")
axs[1][1].plot(x, y_noise, label="Noise")
axs[1][1].set_title("Noise")
axs[1][1].set_xlabel("Timp")
axs[1][1].set_ylabel("Amplitudine")
plt.savefig("8_1_Seria_de_timp.pdf")

# b)

# Corelatia este echivalenta cu convolutia cu array-ul inversat
# Refolosesc implementarea mea din laboratorul 6
def convolutia(polinom1, polinom2):
    N1, N2 = len(polinom1), len(polinom2)
    N = None
    if N1==N2:
        N = N1
    if N1 > N2:
        N = N1
        polinom2 = np.pad(polinom2, (N1-N2, 0))
    else:
        N = N2
        polinom1 = np.pad(polinom1, (N2-N1, 0))
        
    haha = [x for x in range(2*N-1)]
    res = []
    for i in haha:
        sum = 0
        for j in haha[:i+1]:
            if i-j < N and j < N:
                sum += polinom1[j] * polinom2[i-j]
        res.append(sum)
    return np.array(res[2*N-1-(N1+N2)+1:])

# In numpy este functia np.correlate(y, y, mode="full") care este echivalent cu 
# convolutia(y, y[::-1])

y_autocorelatie = convolutia(y,y[::-1])
fig, axs = plt.subplots(2, figsize=(10, 6))
fig.suptitle("Autocorelatia Seriei de Timp")
axs[0].plot(x, y)
axs[0].set_title("Seria de timp")
axs[1].plot(y_autocorelatie)
axs[1].set_title("Autocorelatia")
plt.savefig("8_1_Autocorelatia.pdf")

# c)

def calculate_predictions(serie_timp, orizont_de_timp, p):

    def get_x(serie_timp,p,lungime=None):
        if lungime != None:
            serie_timp = serie_timp[:lungime]
        vector_autocorelatie = convolutia(serie_timp, serie_timp[::-1])
        Gamma_mare = [[0 for _x in range(p)] for _y in range(p)]
        mij = len(vector_autocorelatie)//2
        gamma = vector_autocorelatie[mij:mij+p+1]
        for i in range(p):
            for j in range(p):
                Gamma_mare[i][j] = gamma[abs(j-i)]
        Gamma_mic = gamma[1:]
        Gamma_mare_invers = np.linalg.inv(Gamma_mare)
        x = Gamma_mare_invers @ Gamma_mic
        return x

    x_star = get_x(serie_timp, p)
    Y = [[0 for i in range(p)] for i in range(orizont_de_timp)]
    for u, i in enumerate(range(len(serie_timp)-1, len(serie_timp)-orizont_de_timp-1, -1)):
        for v, j in enumerate(range(p)):
            Y[u][v] = serie_timp[i-j]
    Y = np.array(Y)
    y_caciula = Y @ x_star
    return y_caciula


cutoff_c = 1
y_predictions = np.append(y[:-cutoff_c], calculate_predictions(y[:-cutoff_c], 20, 5)[0])
fig, axs = plt.subplots(2, figsize=(5, 3))
axs[0].plot(y[-10:])
axs[1].plot(y_predictions[-10:])
plt.savefig("8_1_Predicitie la ultimul punct.pdf")

# d)


best_m = 2
best_p = 63
best_MSE = 10e10
cutoff_c = 1

# m = 2 p = 63
# for m_i in range(2, 15):
#     for p_i in range(2, 101):
#         y_prediction = calculate_predictions(y[:-cutoff_c], m_i, p_i)[0]
#         MSE = (y_prediction - y[-1])**2
#         print(f"Incercam m={m_i} p={p_i} cu MSE={MSE}")
#         if MSE < best_MSE:
#             print(f"Success m={m_i} p={p_i}")
#             best_m = m_i
#             best_p = p_i
#             best_MSE = MSE


y_prediction = np.append(y[:-1], calculate_predictions(y[:-cutoff_c], best_m, best_p)[0])
fig, axs = plt.subplots(2, figsize=(8, 10))
fig.suptitle("Predictie dupa fine-tuning")
axs[0].plot(y[-10:])
axs[0].set_title("Original")
axs[0].set_xlabel("Timp")
axs[0].set_ylabel("Amplitudine")
axs[1].plot(y_prediction[-10:])
axs[1].set_title(f"Predicitie (m={best_m}, p={best_p})")
axs[1].set_xlabel("Timp")
axs[1].set_ylabel("Amplitudine")
plt.savefig("8_1_Predictie_Fine_Tunning.pdf")
















    