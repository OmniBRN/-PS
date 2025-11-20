import numpy as np
import matplotlib.pyplot as plt
import scipy as scp


csv = []
with open("/home/tudor/Documents/Anul3/Prelucrarea_Semnalelor/PS/lab6/Train.csv") as file:
    for line in file:
        csv.append(line.split(","))


#a)
x = csv[697:697+72]
x = [int(t[2][:-1]) for t in x]


#b)
w = [5, 13, 17, 57]
W = [np.convolve(x, np.ones(w_x), 'valid') for w_x in w]

#c) 9.92063492063492e-05 - 71.42% din frecventa nyquist si aproximativ odata la 3 ore
# N = len(x)
# t = np.linspace(0, 1/7200, 36)
# t[25]

# x_fft = np.fft.fft(x)
# x_fft = np.abs(x_fft)/N
# x_fft[0] = 0
# x_fft = x_fft[:N//2]

# fig, axs = plt.subplots()
# axs.stem(t, x_fft)
# plt.show()

# d) & e)
ordin = 5
ordin_butter = 3
Wn = 0.7142
rp = 5
butter_b, butter_a = scp.signal.butter(ordin_butter, Wn, btype="low")
cheby_b, cheby_a = scp.signal.cheby1(ordin, rp, Wn, btype="low")

butter_x = scp.signal.filtfilt(butter_b, butter_a, x)
cheby_x = scp.signal.filtfilt(cheby_b, cheby_a, x)

fig, axs = plt.subplots()
axs.plot(x, color="Blue", label="Unfiltered")
axs.plot(butter_x, color="Green", label="Butter")
axs.plot(cheby_x, color="Orange", label="Cheby")
axs.legend()
plt.show()

# f) Pentru a observa trenduri recurente (elimina varfuri particulare) este folositor un 
# filtru butter, cu un ordin mai mic (am vazut ca 3 arata ok) il face sa arate mai consistent
