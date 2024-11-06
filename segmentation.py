import numpy as np
from collections import Counter
from numba import njit
from gaussians import gaussian1d

def regular_segmentation(I, n_tones, vmin=0, vmax=255):
    """
    Apply tone segmentation to the image I.

    Parameters: 
        I: array of int.
        n_tones: int
            Number of tones of the output. We often chose 3 or 4.
        vmin, vmax: int
            Values between wich I is supposed to be. Typically (0,255) but (1,2) in XDoG algorithm.
    """
    return np.floor(I*(n_tones-1)/(vmax-vmin)+.5)*(vmax-vmin)/(n_tones-1)


sup=180

def tore_segmentation(I, n_tones):
    """
    Apply tone segmentation to the image I with values in the torus [0,sup-1].

    Parameters: 
        I: array of int.
        n_tones: int
            Number of tones of the output. We often chose 3 or 4.
    """
    n_tones*=2 #empirical coefficient to be consistent with regular_segmentation
    sigma1=3 #empirical value

    S_max = intermediary(I, sigma1)
    while len(S_max)<n_tones :
            sigma1*=.7 #empirical coefficient
            S_max = intermediary(I, sigma1)
    
    I_segmented = reach_max(I, S_max)
    return I_segmented

def intermediary(I, sigma1):
    """
    Returns the set of most frequent values in the image.
     
    Starts by counting the occurrences of each value. 
    Smoothes the list of occurrences with a Gaussian of standard deviation sigma1. 
    Keeps only the largest peaks in this list.
    """
    L = L_count(I)
    kernel = gaussian1d(sup, std=sigma1)
    L2 = convolve_tore(L, kernel)
    dico_max_L2 = dico_max(L2)
    L_val=np.unique(sorted(dico_max_L2.values()))
    S_max = local_max(L_val, dico_max_L2)
    return S_max

def L_count(I):
    """
    Returns the number of appearances in I of each torus value.
    """
    count = Counter(I.flatten())
    L = [count.get(i, 0) for i in range(sup)]
    return L

@njit
def convolve_tore(L, kernel):
    """
    Returns the convolution between L and kernel, taking into account that L is a torus.
    """
    N = len(L)
    result = np.zeros(N)
    for i in range(N):
        for j in range(N):
            result[i] += L[(i + j - N//2) % N] * kernel[j]
    return result

@njit
def dico_max(L):
    """
    Returns the {index: value} dictionary as soon as L admits a local maximum at this index.
    """
    N = len(L)
    dico_max_L = {}
    for i in range(N):
        if L[i]>L[(i-1)%N] and L[i]>L[(i+1)%N]:
            dico_max_L[i]=L[i]
    return dico_max_L

@njit
def local_max(L_val, dico_max):
    """
    Retains maximum values that are at least .01 of the largest.
    """
    S_max = set()
    for val in L_val[::-1]:
        for key in dico_max.keys():
            if dico_max[key]==val and L_val[-1]/val<100:
                S_max.add(key)
    return S_max

@njit
def reach_max(I, S_max):
    """
    Returns the copy of I where each value has been replaced by the nearest S_max value.
    """
    I_segmented = np.copy(I)
    N,M = I.shape
    for i in range(N):
        for j in range(M):
            p = I[i,j]
            if p not in S_max:
                k=1
                while ((p+k)%sup not in S_max) and ((p-k)%sup not in S_max):
                    k+=1
                if (p+k)%sup in S_max:
                    I_segmented[i,j] = (p+k)%sup
                else:
                    I_segmented[i,j] = (p-k)%sup
    return I_segmented