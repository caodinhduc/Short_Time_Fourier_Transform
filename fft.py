import cmath
import numpy as np
from math import log, ceil


def omega(p, q):
    """ The omega term in DFT formulas"""
    return cmath.exp((2.0 * cmath.pi * 1j * q) / p)


def fft(x):
    """ FFT of 1-d signals
    call : X = fft(x)
    input x = list containing sequences of a discrete time signals
    output X = dft of x 
    """
    n = len(x)
    if n == 1:
        return x
    Feven, Fodd = fft(x[0::2]), fft(x[1::2])
    combined = [0] * int(n)
    for m in range(int(n/2)):
        combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
        combined[m + int(n/2)] = Feven[m] - omega(n, -m) * Fodd[m]
    return combined


def ifft(X):
    """ Inverse FFT of 1-d signals
    call x = ifft(X) 
    unpadding must be done implicitly
    """
    x = fft([x.conjugate() for x in X])
    return [x.conjugate()/len(X) for x in x]


def pad2(x, Nfft):
    
    """ pad zeros to x """
    m, n = np.shape(x)
    F = np.zeros((m, Nfft), dtype = x.dtype)
    F[0:m, 0:n] = x
    print(F.shape)
    return F, m, Nfft


def fft2(f, Nfft=True):
    """ FFT of 2d signals/images with padding
    With each input len n, output len will be n + 1
    usage X, m, n = fft2(x), where m and n are dimensions of original signal
    """
    f, m, n = pad2(f, Nfft)
    result = [fft(a)[: int(n/2+1)] for a in f]
    return np.array(result)
