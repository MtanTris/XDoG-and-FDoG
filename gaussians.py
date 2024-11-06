import numpy as np

def gaussian1d(n, std):
    """
    Returns the central values of a 1D Gaussian in a numpy array.
    
    Parameters:
        n: int
            length of the array
        std: float
            standard deviation of the gaussian
    """
    alpha = 2 * std * std
    logarg = (np.arange(n) - n // 2) ** 2 / alpha
    return np.exp(-logarg) / np.sqrt(np.pi * alpha)

def gaussian2d(m, n, sigma, sigma2):
    """
    Returns the central values of a non-isotropic 2D Gaussian in a numpy array.
    
    Parameters:
        (n,m): (int, int)
            shape of the array
        (sigma, sigma2): (float, float)
            standard deviations of the gaussian in directions x and y
    """
    g1 = gaussian1d(m, sigma)
    g2 = gaussian1d(n, sigma2)
    return np.outer(g1,g2) 