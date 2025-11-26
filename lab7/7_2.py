import scipy as scp
import numpy as np
import matplotlib.pyplot as plt

X = scp.datasets.face(gray=True)
height, width = np.shape(X)

# Varianta animalica fara FFT dar imi dadea foarte gresit cand incercam sa-l fac cu FFT
width, height = np.shape(X)
smoothing_kernel = np.ones((5,5))/25
X_blurred = np.zeros((width+4, height+4))
X_padded = np.pad(X, 2, mode="edge")
for i in range(2, width+2):
    for j in range(2, height + 2):
        sum = 0
        for u_i, u in enumerate(range(i-2, i+3)):
            for v_j, v in enumerate(range(j-2, j+3)):
                sum+= X_padded[u][v] * smoothing_kernel[u_i][v_j]
        X_blurred[u][v] = sum
X_blurred = X_blurred[2:-2, 2:-2] 

norma_X = np.pow(np.linalg.norm(X, 2),2)
norma_X_blurred = np.pow(np.linalg.norm(X - X_blurred, 2),2)

SNR = 10 * np.log10(norma_X/norma_X_blurred)
print(SNR)

fig, axs = plt.subplots(1, 3, figsize=(10,5))
axs[0].imshow(X, cmap=plt.cm.gray)
axs[0].set_title("Imaginea Originala")
axs[1].imshow(X_blurred, cmap=plt.cm.gray)
axs[1].set_title("Imaginea Blurata (SNR = ~25)")
axs[2].imshow(X-X_blurred, cmap=plt.cm.gray)
plt.savefig("2.pdf")