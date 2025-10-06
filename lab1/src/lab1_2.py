import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')



# 2.a)
Ts1 = 1/400
t1 = np.linspace(0, 1, 1600)
x = np.sin(2*np.pi*1/Ts1*t1)
fig, axs = plt.subplots()
fig.suptitle("Semnal Sinusoidal de Frecventa 400Hz")
plt.xlabel("Timp")
plt.ylabel("Semnal")
axs.plot(t1,x)
fig.savefig("2_400Hz.pdf")
plt.show()

# 2.b)
Ts2 = 1/800
t2 = np.linspace(0, 3, 3*4800)
y = np.sin(2*np.pi*1/Ts2*t2)
fig, axs = plt.subplots()
fig.suptitle("Semnal Sinusoidal de Frecventa 800Hz care sa dureze 3 secunde")
plt.xlabel("Timp")
plt.ylabel("Semnal")
axs.plot(t2,y)
fig.savefig("2_800Hz3s.pdf")
plt.show()

# 2.c)
t3 = np.linspace(0, 1, 960)
z = np.mod(t3, 1/240)
fig, axs = plt.subplots()
fig.suptitle("Semnal Sawtooth de frecventa 240Hz")
plt.xlabel("Timp")
plt.ylabel("Semnal")
axs.plot(t3,z)
fig.savefig("2_Sawtooth240Hz.pdf")
plt.show()

# 2.d)
t4 = np.linspace(0, 1, 1200)
t = np.sign(np.sin(2*np.pi*t4*300))
fig, axs = plt.subplots()
fig.suptitle("Semnal Square de frecventa 300Hz")
plt.xlabel("Timp")
plt.ylabel("Semnal")
axs.plot(t4,t)
fig.savefig("2_Square300Hz.pdf")
plt.show()

# 2.e)
t5 = np.random.rand(128,128)
plt.imshow(t5)
plt.title("Semnal 2D aleator")
plt.savefig("2_Semnal2Daleator.pdf")
plt.show()

# 2. f)
def shgradient():
    matrix = np.empty((128,128))
    for i in range(128):
        for j in range(128):
            matrix[i,j] = min(np.sqrt(i*i+j*j), np.sqrt((128-i)*(128-i)+(128-j)*(128-i)))/(128 * np.sqrt(2))
    return matrix
    

t6 = shgradient()
plt.imshow(t6)
plt.title("Semnal 2D - Gradient")
plt.savefig("2_Semnal2Dgradient.pdf")
plt.show()
