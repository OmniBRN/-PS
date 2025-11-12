import numpy as np
import matplotlib.pyplot as plt

# a) Care este frecventa de esantionare a semnalului din Train.csv (revedeti sectiunea pentru detalii despre cum a fost achizitionat acesta)?
#   fs = 1/3600 
#    
# b) Ce interval de timp acopera esantioanele din fisier?
#   25-08-2012 00:00 - 25-09-2014 23:00 (18288 ore)
# c) Considerand ca semnalul a fost esantionat corect (fara aliere) si optim, care este frecventa maxima prezenta ın semnal?
#  66.85 * 1/3600 = 0.0185 Hz

# d)

x_ore = np.genfromtxt('/home/tudor/Documents/Anul3/Prelucrarea_Semnalelor/PS/lab5/bullettrain-timeseries-data/Train.csv', delimiter=',')
x_ore = np.array([t[2] for t in x_ore[1:]])
# e) Da, semnalul prezinta o componenta continua.
# Am determinat asta afland media vectorului si urmarind graficul dupa transformata fourier (un comportament asemanator ca in figura 1)
# Am eliminat componenta continua scazand media vectorului din fiecare element, ducand media la 0

x_ore_2 = x_ore - np.mean(x_ore)
N = len(x_ore_2)
X = np.fft.fft(x_ore_2)
X = X[:N//2]
X = abs(X/N)
X_2 = np.array([(x, i) for i, x in enumerate(X)])


top = np.array([y[1] for y in (sorted(X_2, key=lambda x:x[0] ,reverse=True)[:4])])
print(top)
f = 1/3600 * np.linspace(0, N//2, N//2)

fig, axs = plt.subplots()
axs.plot(f, X)
plt.show()

# f) 
x_ore_luna = x_ore[2736:3480]

fig, axs = plt.subplots()
axs.plot(x_ore_luna)
plt.show()

