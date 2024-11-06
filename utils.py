import cv2
import numpy as np
from segmentation import tore_segmentation, regular_segmentation


def apply_to_components(I, func, base, n_tones, *args):
    """
    Apply func to each component of I.

    Parameters:
        I: array of dimension 2 or 3
            Picture on which func will be applied.
        func: function
            In practice, the function used is XDoG or FDoG.
        base: string
            The basis for breaking down the image into its component parts.
            "gray" is used to work in gray scale,
            "RGB" or "HSV" are the two authorized three-dimensional bases.
        n_tones: int
            The number of tones used in the segmentation of suitable components.
        *args:
            Parameters used for each application of func.
    """
    if I.ndim > 3:
        I = I[:,:,:3]
    I = 255*normalise(I)
    if base == 'gray':
        if I.ndim == 3: 
            I = .2989*I[:,:,0] + .5870*I[:,:,1] + .1140*I[:,:,2]
        output = func(I, *args, n_tones=n_tones)
    elif base == 'RGB':
        R, G, B = [func(I[:,:,i], *args, n_tones=n_tones) for i in range(3)]
        output = np.stack((R, G, B), axis=-1)
    elif base == 'HSV':
        I2 = cv2.cvtColor(cv2.convertScaleAbs(I), cv2.COLOR_RGB2HSV).astype(np.float64)
        if n_tones is None:
            H, S = I2[:,:,0], I2[:,:,1]
        else:
            H = tore_segmentation(I2[:,:,0], n_tones)
            S = regular_segmentation(I2[:,:,1], n_tones)
        V = func(I2[:,:,2], *args, n_tones=n_tones)
        O = np.stack((H, S, V), axis=-1)
        output = cv2.cvtColor(cv2.convertScaleAbs(O), cv2.COLOR_HSV2RGB).astype(np.float64)
    else:
        raise ValueError("Incorrect value for base. Please use 'gray', 'RGB' or 'HSV'.")
    return output.astype(np.uint8)


def normalise_2d(I_2d, I_2d_min_max):
    """
    Normalises I_2d between 0 and 1

    Parameters:
        I_2d: numpy two-dimensionnal array
        I_2d_min_max: tuple or NoneType
            If None is used, the values will the minimum and maximum of I_2d.
            If a tuple is given, its values are the one between wich I_2d is supposed to be.
    """
    if I_2d_min_max is None:
        I_2d_min_max = (np.min(I_2d), np.max(I_2d))
    I_2d_min, I_2d_max = I_2d_min_max
    assert I_2d_max != I_2d_min, ("Normalisation asked for a constant vector")
    return (I_2d - I_2d_min)/(I_2d_max - I_2d_min)

def normalise(I, I_min_max=None):
    """
    Normalises I between 0 and 1. 

    Parameters:
        I: numpy array of dimension 2 or 3.
        I_min_max: tuple or NoneType
            If None is used, the values will the minimum and maximum of I.
            If a tuple is given, its values are the one between wich I is supposed to be.
    """
    I2 = np.array(I, dtype=np.float64)
    if I2.ndim == 3:
        for i in range(I.shape[2]):
            I2[:,:,i] = normalise_2d(I2[:,:,i], I_2d_min_max=I_min_max)
    else: #I2.ndim=2
        I2 = normalise_2d(I2, I_2d_min_max=I_min_max)
    return I2


def handle_borders(u):
    """
    Multiplies image size by 2 in each direction to manage edge effects.
    """
    u = np.array(u)
    N, M = u.shape
    blank = np.zeros((2*N, 2*M))
    reversed_x = u[::-1]
    reversed_y = u[:, ::-1]
    reversed_xy = reversed_x[:, ::-1]
    blank[:N, :M] = u
    blank[N:, M:] = reversed_xy
    blank[:N, M:] = reversed_y
    blank[N:, :M] = reversed_x
    return blank

