from scipy import misc, ndimage
import numpy as np
import matplotlib.pyplot as plt
size = 100

x_1 = np.fromfunction(lambda i, j: np.sin(2*np.pi*i + 3*np.pi*j), (size, size))
X_1 = np.fft.fft2(x_1)
X_1_db = np.log10(np.abs(X_1))

# sin(4pi * i) = sin(2pi * 2*i) = 0
# cos(6pi * j) = cos(2pi * 3*i) = 1
# deci defapt x_2 este o matrice care contine doar 1
#x_2 = np.fromfunction(lambda i, j: np.sin(4*np.pi*i) + np.cos(6*np.pi*j), (size, size))
x_2 = np.ones((size, size))
X_2 = np.fft.fft2(x_2)
X_2_abs = np.abs(X_2)

Y_1 = np.zeros((size, size))
Y_1[0][5], Y_1[0][(size-1)-5] = 1,1
y_1 = np.abs(np.fft.ifft2(Y_1))

Y_2 = np.zeros((size, size))
Y_2[5][0], Y_2[(size-1)- 5][0]  = 1, 1
y_2 = np.abs(np.fft.ifft2(Y_2))

Y_3 = np.zeros((size, size))
Y_3[5][5], Y_3[(size-1)- 5][(size-1)-5]  = 1, 1 
y_3 = np.abs(np.fft.ifft2(Y_3))

# plt.imshow(x_1)
# plt.colorbar()
# plt.savefig("1_x_1.pdf")

# plt.imshow(X_1_db)
# plt.colorbar()
# plt.savefig("1_X_1.pdf")

# plt.imshow(x_2)
# plt.colorbar()
# plt.savefig("1_x_2.pdf")

# plt.imshow(X_2_abs)
# plt.colorbar()
# plt.savefig("1_X_2.pdf")

# plt.imshow(y_1)
# plt.colorbar()
# plt.savefig("1_y_1.pdf")

# plt.imshow(Y_1)
# plt.colorbar()
# plt.savefig("1_Y_1.pdf")

# plt.imshow(y_2)
# plt.colorbar()
# plt.savefig("1_y_2.pdf")

# plt.imshow(Y_2)
# plt.colorbar()
# plt.savefig("1_Y_2.pdf")

# plt.imshow(y_3)
# plt.colorbar()
# plt.savefig("1_y_3.pdf")

# plt.imshow(Y_3)
# plt.colorbar()
# plt.savefig("1_Y_3.pdf")