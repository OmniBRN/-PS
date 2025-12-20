import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
np.random.seed(25)

x = np.linspace(-5, 5, 1000)
y_trend = x**2 / 20
y_sezon =  np.sin(2 * np.pi * 15 * x + np.pi/4) + 1/2 * np.sin(2 * np.pi * 6 * x)
y_noise = np.random.normal(size=1000)
y = y_trend + y_sezon + y_noise

q = 15
number_of_predictions = 10
epsilon = y - np.mean(y)

def get_theta(serie_timp, q):
    N = len(serie_timp)
    media = np.mean(serie_timp)
    epsilon = serie_timp - media
    epsilon_trunchiat = epsilon[q:]
    Y = np.zeros((N-q, q))
    for i in range(N-q):
        Y[i, :] = epsilon[i: i+q][::-1]
    
    theta, _, _, _ = np.linalg.lstsq(Y, epsilon_trunchiat, rcond=None)
    return theta, media

def make_predictions_MA(number_of_predictions, serie_de_timp, q, theta, media):
    for _ in range(number_of_predictions):
        epsilon = serie_de_timp - media
        epsilon = epsilon[-q:][::-1]
        predictie = media + np.dot(epsilon,theta)
        serie_de_timp = np.append(serie_de_timp, predictie)
    return serie_de_timp

y_predictions = y[:-number_of_predictions]
theta, media = get_theta(y_predictions, q)
y_predictions = make_predictions_MA(number_of_predictions, y_predictions,q, theta, media)


fig, axs = plt.subplots(2, figsize=(8, 10))
fig.suptitle("Model MA - q = 15")
axs[0].plot(x[-15:], y[-15:])
axs[0].set_title("Seria originala")
axs[0].set_ylabel("Amplitudine")
axs[0].set_xlabel("timp")
axs[1].plot(x[-15:], y_predictions[-15:])
axs[1].set_ylabel("Amplitudine")
axs[1].set_xlabel("timp")
axs[1].set_title("Seria dupa prezicere")
plt.savefig("9_3_Predictie_model_MA.pdf")



    

    








