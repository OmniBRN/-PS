import scipy.io
import sounddevice as sd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')

t = np.linspace(0,1, 20000)
x1 = np.sign(np.sin(2*np.pi * 400 * t))
x2 = np.sign(np.sin(2*np.pi * 675 * t))

x = np.concatenate((x1,x2))
sd.play(x, 44100)
scipy.io.wavfile.write("2_5.wav", 44100, x)

# Frecventa mai mica produce un sunet mai gros si frecventa mai mare produce un sunt mai subtire

