import numpy as np
import matplotlib.pyplot as plt

# a) Care este frecventa de esantionare a semnalului din Train.csv (revedeti sectiunea pentru detalii despre cum a fost achizitionat acesta)?
#   fs = 1/3600 
#    
# b) Ce interval de timp acopera esantioanele din fisier?
#   25-08-2012 00:00 - 25-09-2014 23:00 (18288 ore)
# c) Considerand ca semnalul a fost esantionat corect (fara aliere) si optim, care este frecventa maxima prezenta ın semnal?

# d)

x_ore = np.genfromtxt('/home/tudor/Documents/Anul3/Prelucrarea_Semnalelor/PS/lab5/bullettrain-timeseries-data/Train.csv', delimiter=',')
x_ore = np.array([t[2] for t in x_ore[1:]])
# e) Da, semnalul prezinta o componenta continua.
# Am determinat asta afland media vectorului si urmarind graficul dupa transformata fourier (un comportament asemanator ca in figura 1)
# Am eliminat componenta continua scazand media vectorului din fiecare element, ducand media la 0
x_ore_2 = x_ore - np.mean(x_ore)
N = len(x_ore_2)
X = np.fft.fft(x_ore_2)
X_vechi = X
X = X[:N//2]
X = abs(X/N)
X_2 = np.array([(x, i) for i, x in enumerate(X)])

f = 1/3600 * np.linspace(0, N//2, N//2)
fig, axs = plt.subplots(2, figsize=(6,8))
axs[0].plot(f, X)
axs[1].plot(x_ore_2)
fig.suptitle("d) FFT pe setul de date")
axs[0].set_ylabel("|X|")
axs[0].set_xlabel("Frecventa (Hz)")
axs[1].set_ylabel("Numar Vanzari")
axs[1].set_xlabel("Timp (h)")
plt.savefig("FFT.pdf")
plt.show()

# f) Primele 4 cele mai mari valori ale absolutei, cu frecventele corespunzatoare
# #1: 0.0002777777777777778 Hz - 66.85385766393449
#  se intampla odata la 3600.0 secunde -> Se vinde o masina odata la o ora
# #2: 0.0005555555555555556 Hz - 35.2191729779369
#  se intampla odata la 1800.0 secunde -> Se vinde o masina odata la jumatate de ora
# #3: 0.21166666666666667 Hz - 27.10202228761556
#  se intampla odata la 4.724409448818897 secunde -> huh
# #4: 0.0008333333333333334 Hz - 25.219916484044816
#  se intampla odata la 1200.0 secunde -> Se vine o masina odata la 20 de minute
top = np.array([(y[1]/3600, y[0]) for y in (sorted(X_2, key=lambda x:x[0] ,reverse=True)[:4])])
for i in range(4):
    print(f"#{i+1}: {top[i][0]} Hz - {top[i][1]}")
    print(f" se intampla odata la {1/top[i][0]} secunde") 


# f) 
x_ore_luna = x_ore[2736:3480]
fig, axs = plt.subplots()
fig.suptitle("Perioada de timp 17-12-2012 - 17-01-2013")
axs.plot(x_ore_luna)
axs.set_ylabel("Vanzari de masini")
axs.set_xlabel("Timp (h)")
plt.savefig("Luna.pdf")
plt.show()

# h) Factorii prin care putem afla data la care incepe esantionarea 
# sunt urmatorii:
# h.1) La nivel de ore, la inceputul zilei este mai multa activitate dimineata decat seara
# h.2) La nivel de saptamana depinde mult despre ce vorbim, la masini nu pare sa fie un 
# pattern desi pare ca luni se intampla cele mai multe vanzari
# h.3) La nivel de luni, majoritatea vanzarilor se intampla la inceput de luna 
# h.4) La nivel de an, cele mai multe vanzari de masini se intampla in timpul verii si
# finalul anului deoarece atunci apar promotiile de sarbatori

# i) Am filtrat primele 4 cele mai mari frecvente si acuma am...
X_vechi = [x if np.abs(x)/N < 15 else 0 for x in X_vechi]
x = np.fft.ifft(X_vechi)
X_vechi = X_vechi[:N//2]
X_vechi = np.abs(X_vechi)/N

fig, axs = plt.subplots(2, figsize=(6,8))
fig.suptitle("i) Filtrat frecventele cu |X[omega]| > 15")
axs[0].plot(f, X_vechi)
axs[0].set_xlabel("Frecvente (Hz)")
axs[0].set_ylabel("|X|")
axs[1].plot(x)
axs[1].set_xlabel("Timp (h)")
axs[1].set_ylabel("Numarul de masini vandute")
plt.savefig("Filtrat.pdf")
plt.show()

