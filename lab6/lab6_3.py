import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

# def produsul_polinoamelor(polinom1, polinom2):
#     # Am zis sa o fac cum as face o inmultire de polinoame pe foaie
#     polinom1 = [(x, len(polinom1)-i-1) for i, x in enumerate(polinom1)]
#     polinom2 = [(x, len(polinom2)-i-1) for i, x in enumerate(polinom2)]
#     dic = {}
#     for i1 in polinom1:
#         for i2 in polinom2:
#             putere = i1[1] + i2[1]
#             if putere in dic.keys():
#                 dic[putere] += i1[0]  * i2[0]
#             else:
#                 dic[putere] = i1[0] * i2[0]
#     res = [dic[x] for x in dic.keys()]
#     return np.array(res)

def convolutia(polinom1, polinom2):
    N1, N2 = len(polinom1), len(polinom2)
    N = None
    if N1==N2:
        N = N1
    if N1 > N2:
        N = N1
        polinom2 = np.pad(polinom2, (N1-N2, 0))
    else:
        N = N2
        polinom1 = np.pad(polinom1, (N2-N1, 0))
        
    haha = [x for x in range(2*N-1)]
    res = []
    for i in haha:
        sum = 0
        for j in haha[:i+1]:
            if i-j < N and j < N:
                sum += polinom1[j] * polinom2[i-j]
        res.append(sum)
    return np.array(res[2*N-1-(N1+N2)+1:])

def convolutie_fft(polinom1, polinom2):
    N1, N2 = len(polinom1), len(polinom2)
    N = None
    if N1==N2:
        N = N1
    if N1 > N2:
        N = N1
        polinom2 = np.pad(polinom2, (N1-N2, 0))
    else:
        N = N2
        polinom1 = np.pad(polinom1, (N2-N1, 0))
    p1_fft = np.fft.fft(np.pad(polinom1, (0,N-1)))
    p2_fft = np.fft.fft(np.pad(polinom2, (0,N-1)))
    prod =  p1_fft * p2_fft
    res = np.fft.ifft(prod)
    res = np.real(res)
    return np.array(res[2*N-1-(N1+N2)+1:])

p1 = [33, 5, -128]
p2 = [-13, 4, -8, 17]
print(convolutia(p1,p2))
print(convolutie_fft(p1,p2))
# Sanity Check
print(scp.signal.fftconvolve(p1,p2))

