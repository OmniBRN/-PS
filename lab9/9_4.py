import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
np.random.seed(25)

x = np.linspace(-5, 5, 1000)
y_trend = x**2 / 20
y_sezon =  np.sin(2 * np.pi * 15 * x + np.pi/4) + 1/2 * np.sin(2 * np.pi * 6 * x)
y_noise = np.random.normal(size=1000)
y = y_trend + y_sezon + y_noise

p_max = 10
q_max = 10
best_aic = float("inf")
best_combo = None
for p in range(2, p_max):
    for q in range(2, q_max):
        model = ARIMA(y[:-10], order=(p, 0, q)).fit()
        if model.aic < best_aic:
            best_aic = model.aic
            best_combo = (p, q)

best_model = ARIMA(y[:-10], order=(best_combo[0],0, best_combo[1])).fit()
y_predictions = np.append(y[:-10],best_model.forecast(steps=10))


fig, axs = plt.subplots(2, figsize=(8, 10))
fig.suptitle(f"Model ARMA - p = {best_combo[0]} q = {best_combo[1]}")
axs[0].plot(x[-15:], y[-15:])
axs[0].set_title("Seria originala")
axs[0].set_ylabel("Amplitudine")
axs[0].set_xlabel("timp")
axs[1].plot(x[-15:], y_predictions[-15:])
axs[1].set_ylabel("Amplitudine")
axs[1].set_xlabel("timp")
axs[1].set_title("Seria dupa prezicere")
plt.savefig("9_4_Predictie_model_ARMA.pdf")











