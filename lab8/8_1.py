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

y_norm = y - np.mean(y)
y_autocorelatie = convolutia(y_norm,y_norm[::-1])
y_autocorelatie = y_autocorelatie / np.max(y_autocorelatie)
fig, axs = plt.subplots(2, figsize=(10, 6))
fig.suptitle("Autocorelatia Seriei de Timp")
axs[0].plot(x, y)
axs[0].set_title("Seria de timp")
axs[1].plot(y_autocorelatie)
axs[1].set_title("Autocorelatia")
plt.savefig("8_1_Autocorelatia.pdf")

# c)

def get_x(serie_timp,p):

    serie_timp = serie_timp - np.mean(serie_timp)
    vector_autocorelatie = convolutia(serie_timp, serie_timp[::-1])
    mij = len(vector_autocorelatie)//2
    gamma = vector_autocorelatie[mij: mij+p+1]
    if gamma[0] != 0:
        temp = gamma[0]
        gamma = gamma / temp
    Gamma_mare = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            Gamma_mare[i][j] = gamma[abs(j-i)]
    gamma_mic = gamma[1:p+1]
    x = np.linalg.solve(Gamma_mare, gamma_mic)
    return x

def make_predictions(number_of_predictions, serie_timp, p, x_star):
    for _ in range(number_of_predictions):
        predictie = np.dot(serie_timp[-p:][::-1] - medie, x_star)
        serie_timp = np.append(serie_timp, predictie + medie)
    return serie_timp


m = 10
p = 7
number_of_predictions = 10
x_star = get_x(y[-m:], p)
medie = np.mean(y[-m:])

y_predictions = y[:-number_of_predictions]
y_predictions = make_predictions(number_of_predictions, y_predictions, p, x_star)

fig, axs = plt.subplots(2, figsize=(8, 10))
axs[0].plot(x[-15:], y[-15:])
axs[0].set_title("Seria originala")
axs[0].set_ylabel("Amplitudine")
axs[0].set_xlabel("timp")
axs[1].plot(x[-15:], y_predictions[-15:])
axs[1].set_ylabel("Amplitudine")
axs[1].set_xlabel("timp")
axs[1].set_title("Seria dupa prezicere")
plt.savefig("8_1_Predicitie la ultimul punct.pdf")

# # d)


best_MSE = 10e10
best_prediction = None
for p_i in range(2, 50):
    for m_i in range(p_i+2, 251):
        number_of_predictions = 1
        x_star = get_x(y[-m_i:], p_i)
        medie = np.mean(y[-m_i:])
        y_predictions = y[:-number_of_predictions]
        y_prediction = make_predictions(number_of_predictions, y_predictions, p_i, x_star)[-1]
        MSE = (y_prediction - y[-1])**2
        print(f"Incercam m={m_i} p={p_i} cu MSE={MSE}")
        if MSE < best_MSE:
            print(f"Success m={m_i} p={p_i}")
            best_m = m_i
            best_p = p_i
            best_MSE = MSE
            best_prediction = y_prediction


y_predictions = np.append(y[:-1], best_prediction)
print(best_m, best_p, best_MSE)
fig, axs = plt.subplots(2, figsize=(8, 10))
fig.suptitle("Predictie dupa fine-tuning")
axs[0].plot(x[-5:], y[-5:])
axs[0].set_title("Original")
axs[0].set_xlabel("Timp")
axs[0].set_ylabel("Amplitudine")
axs[1].plot(x[-5:], y_predictions[-5:])
axs[1].set_title(f"Predicitie (m={best_m}, p={best_p})")
axs[1].set_xlabel("Timp")
axs[1].set_ylabel("Amplitudine")
plt.savefig("8_1_Predictie_Fine_Tunning.pdf")


