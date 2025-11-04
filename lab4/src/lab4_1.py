import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import scipy as sp
import math
import time

def case(n, numpy_FFT_case = False, FFT_case=False, DFT_case=False):
    t = np.linspace(0,1,n)
    f = np.sin(2*np.pi*t*16) + np.sin(2*np.pi*t*22)
    if FFT_case == True:
        start_time = time.time()
        FFT_f = FFT(f)
        end_time = time.time()
        # print(f"FFT - {n} - {end_time-start_time}")
        return end_time-start_time
    if DFT_case == True:
        start_time = time.time()
        fourier_matrix = getFourierMatrix(n)
        DFT_f = fourier_matrix @ f
        end_time = time.time()
        # print(f"DFT - {n} - {end_time-start_time}")
        return end_time-start_time
    # print(np.allclose(np.fft.fft(f), FFT_f))
    if numpy_FFT_case == True:
        start_time = time.time()
        result = np.fft.fft(f)
        end_time = time.time()
        return end_time-start_time


def FFT(x):
    N = len(x)
    if N == 1:
        return x
    else:
        result_1 = FFT(x[::2])
        result_2 = FFT(x[1::2])
        factor = math.e**(-2j*np.pi*np.arange(N)/N)
        X = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            X[k] = result_1[k] + factor[k] * result_2[k]
            X[k + N // 2] = result_1[k] - factor[k] * result_2[k]
        return X


def getFourierMatrix(order=8):
    result = []
    for i in range(0, order):
        t = np.array([])
        for j in range(0, order):
            t = np.append(t, math.e**(-2j*np.pi*i*j/order))
        result.append(t.copy())
    return np.array(result)

cases = [128, 256, 512, 1024, 2048, 4096, 8192]
results_FFT = [case(x, FFT_case=True) for x in cases]
# results_DFT = [case(x, DFT_case=True) for x in cases] - Le-am calculat odata nu le calculez de fiecare data cand rulez asta
results_DFT = [0.03871583938598633, 0.19036269187927246, 0.6676685810089111, 2.716001033782959, 11.538421869277954, 51.708831787109375, 252.47662734985352]
results_np_FFT = [case(x, numpy_FFT_case=True) for x in cases]
print(results_np_FFT)


fig, axs = plt.subplots()
fig.suptitle("Exercitiul 1")
axs.set_yscale('log')
axs.set_xlabel("N")
axs.set_ylabel("Timp(s)")
line_dft , = axs.plot(cases, results_DFT)
line_dft.set_label("DFT")
line_fft , = axs.plot(cases, results_FFT)
line_fft.set_label("FFT")
line_np_fft , = axs.plot(cases, results_np_FFT)
line_np_fft.set_label("Numpy FFT")
axs.legend()
plt.savefig("4_1.pdf")
