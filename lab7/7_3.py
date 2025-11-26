import scipy as scp
import numpy as np
import matplotlib.pyplot as plt

pixel_noise = 200
X = scp.datasets.face(gray=True)
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

norma_X = np.pow(np.linalg.norm(X, 2),2)
norma_X_noisy = np.pow(np.linalg.norm(X - X_noisy, 2),2)

SNR = 10 * np.log10(norma_X/norma_X_noisy)
print(f"SNR inainte:{SNR}")

# Varianta animalica fara FFT dar imi dadea foarte gresit cand incercam sa-l fac cu FFT
def blurezImaginea(X_noisy):
    width, height = np.shape(X)
    smoothing_kernel = np.ones((5,5))/25
    X_noisy_blurred = np.zeros((width+4, height+4))
    X_noisy_padded = np.pad(X_noisy, 2, mode="edge")
    for i in range(2, width+2):
        for j in range(2, height + 2):
            sum = 0
            for u_i, u in enumerate(range(i-2, i+3)):
                for v_j, v in enumerate(range(j-2, j+3)):
                    sum+= X_noisy_padded[u][v] * smoothing_kernel[u_i][v_j]
            X_noisy_blurred[u][v] = sum
    X_noisy_blurred = X_noisy_blurred[2:-2, 2:-2] 
    return X_noisy_blurred
X_noisy_blurred = blurezImaginea(X_noisy)
X_noisy_blurred = blurezImaginea(X_noisy)
norma_X_noisy_blurred = np.pow(np.linalg.norm(X - X_noisy_blurred, 2),2)
SNR = 10 * np.log10(norma_X/norma_X_noisy_blurred)

print(f"SNR dupa:{SNR}")

fig, axs = plt.subplots(1, 3, figsize=(10,5))
axs[0].set_title("Imaginea originala")
axs[0].imshow(X, cmap=plt.cm.gray)
axs[1].set_title("Imaginea cu noise")
axs[1].imshow(X_noisy, cmap=plt.cm.gray)
axs[2].set_title("Imaginea cu noise blurata")
axs[2].imshow(X_noisy_blurred, cmap=plt.cm.gray)
plt.savefig("3.pdf")