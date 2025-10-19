import scipy.io
import scipy.signal
import sounddevice as sd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

rate = 44100
# a)
Ts1 = 1/400
t1 = np.linspace(0, 1, 16000)
x = np.sin(2*np.pi*1/Ts1*t1)
sd.play(x, 44100)
scipy.io.wavfile.write("2_3_a.wav", rate, x)

# b
Ts2 = 1/800
t2 = np.linspace(0, 3, 3*4800)
y = np.sin(2*np.pi*1/Ts2*t2)
sd.play(y, 44100)
scipy.io.wavfile.write("2_3_b.wav", rate, y)

# c 
t3 = np.linspace(0, 1, 9600*2)
z = 240 * np.mod(t3, 1/240)
sd.play(z, 44100)
scipy.io.wavfile.write("2_3_c.wav", rate, z)

#d
t4 = np.linspace(0, 1, 12000)
t = np.sign(np.sin(2*np.pi*t4*300))
sd.play(t, 44100)
scipy.io.wavfile.write("2_3_d.wav", rate, t)