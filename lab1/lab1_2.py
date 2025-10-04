import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('qtAgg')



# 2.a)
Ts1 = 1/400
t1 = np.linspace(0, 1, 1600)
x = np.sin(2*np.pi*1/Ts1*t1)
fig, axs = plt.subplots()
axs.plot(t1,x)
plt.show()

# 2.b)
Ts2 = 1/800
t2 = np.linspace(0, 3)
y = np.sin(2*np.pi*1/Ts2*t2)
fig, axs = plt.subplots()
axs.plot(t2,y)
plt.show()

# 2.c)
t3 = np.linspace(0, 3, 720)
z = np.mod(t3, 1)
fig, axs = plt.subplots()
axs.plot(t3,z)
plt.show()

# 2.d)
t4 = np.linspace(0, 3, 300)
t = np.sign(np.sin(2*np.pi*t4*300))
fig, axs = plt.subplots()
axs.plot(t4,t)
plt.show()

# 2.e)
t5 = np.random.rand(128,128)
plt.imshow(t5)
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
plt.show()


# fig, axs = plt.subplots(4)
# fig.suptitle('test')
# axs.plot(t1,x)
# axs.plot(t2,y)
# axs.plot(t3,z)
# axs.plot(t4,t)

# plt.show()

