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


def pad1(x, fft_size):
    """ padding for 1d signal
        Add zeros if fft_size is larger than x, else truncate
    """
    m = x.shape[0]
    if m >= fft_size:
        return x[0: fft_size]
    F = np.zeros(fft_size, dtype=x.dtype)
    F[0: m] = x
    return F


def fft_1d(x, fft_size):
    """ Fast Fourier Transfrom for 1d signal
        fft_size is the length of FFT used
    """
    x = pad1(x, fft_size)
    return np.array(fft(x))
    
    
def pad2(x, fft_size):
    
    """ pad zeros to 2d signal """
    m, n = np.shape(x)
    if n >= fft_size:
        return x[0: m, 0: fft_size], m, fft_size
    
    F = np.zeros((m, fft_size), dtype = x.dtype)
    F[0:m, 0:n] = x
    return F


def fft_2d(x, fft_size):
    """ FFT of 2d signals/images with fft_size
    With fft_size is n, length of FFT used is n + 1
    """
    f = pad2(x, fft_size)
    result = [fft(a)[: int(fft_size/2+1)] for a in f]
    return np.array(result)
