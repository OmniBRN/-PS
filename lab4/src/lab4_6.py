import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math
from scipy.io import wavfile

samplerate, data = wavfile.read('/home/tudor/Documents/Anul3/Prelucrarea_Semnalelor/PS/lab4/src/aeiou.wav')
print(len(data))
data = np.array([(x[1]-x[0])//2 + x[0] for x in data])
# wavfile.write("test.wav", samplerate, data)
groups = []
data_length = len(data)
groupSize = data_length//100
jumpSize = groupSize // 2
start = 0
while start+groupSize<=data_length:
    print(f"{start} - {start+groupSize}")
    strip = data[start:start+groupSize]
    groups.append(strip)
    start += jumpSize

FFTs = np.abs(np.array([np.fft.fft(x) for x in groups]))

# FFTs = 20* np.log10(FFTs+1e-10)
# vmax = FFTs.max()
# FFTs = np.clip(FFTs, vmax-80,vmax)
plt.imshow(FFTs, cmap="magma", origin="lower", aspect="auto")
plt.ylabel("Timp (s)")
plt.savefig("4_4.pdf")
