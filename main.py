# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:25:32 2024

@author: trist
"""

from utils import handle_borders, normalise
from gaussians import gaussian2d
from numpy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt
from xdog import xdog
from fdog import fdog
from tqdm import tqdm
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter


def motion_blur(I, sigma, vertical=False):
    """
    Blur the given picture in one direction, creating an illusion of movement.
    This function is mainly used in speed_lines().

    Parameters
    ----------
    I : 2d-array or 3d-array
        Data about the color of each pixel in a picture.
    sigma : float
        Standard deviation of the blur effect.
        Use a higher value to make the motion look faster.
    vertical : bool, optional
        Precise if the movement should be vertical instead of horizontal. 
        The default is False.

    Returns
    -------
    2d-array
        Blurred image.

    """
    if I.ndim == 3: 
        I = .2989*I[:,:,0] + .5870*I[:,:,1] + .1140*I[:,:,2]
    N, M = I.shape
    I = handle_borders(I)
    if vertical:
        gauss = gaussian2d(2*N, 2*M, sigma, 1e-10)
    else:
        gauss = gaussian2d(2*N, 2*M, 1e-10, sigma)
    N, M = I.shape
    blurred = np.real(ifft2(fft2(gauss)*fft2(I)))[N//2:, M//2:]
    return normalise(blurred)


def speed_lines(I, sigma_v=25, sigma=.7, p=65, epsilon=45, phi=.015, vertical=False):
    """
    Use the XDoG algorithm to create a speed-lines effect on an image, namely
    the effect used in comics/manga to create the illusion of a fast motion.

    Parameters
    ----------
    I : 2d-array or 3d-array
        Data about the color of each pixel in a picture.
    sigma_v : float, optional
        Describe the standard deviation of the blurring effect. 
        Use a higher value to make the motion look faster. The default is 25.
    sigma : float, optional
        See the XDoG documentation for more information. The default is .7.
    p : float, optional
        See the XDoG documentation for more information. The default is 65.
    epsilon : TYPE, optional
        See the XDoG documentation for more information. The default is 45.
    phi : TYPE, optional
        See the XDoG documentation for more information. The default is .015.
    vertical : TYPE, optional
        Precise if the movement should be vertical instead of horizontal. 
        The default is False.

    Returns
    -------
    2d-array
        Modified image with added style effect.

    """
    if I.ndim == 3: 
        I = .2989*I[:,:,0] + .5870*I[:,:,1] + .1140*I[:,:,2]
    blurred = normalise(motion_blur(I, sigma_v, vertical=vertical))
    return xdog(blurred, sigma, p, epsilon, phi)


def textures(I, sigmas=None, ps=None, textures=None, negatives=None):
    """
    Use the XDoG algorithm to apply textures locally to an image, depending on
    the initial contrast.
    Use the default parameters as an example.

    Parameters
    ----------
    I : 2d-array or 3d-array
        Data about the color of each pixel in a picture.
    sigmas : tuple of floats, optional
        Describe the standard deviation used to apply each texture. 
        Use a lesser value to apply the texture more locally. 
        The default is None.
    ps : tuple of floats, optional
        Describe the sharpness with which each texture will be applied. 
        Use a higher value to make the texture apply to more details.
        The default is None.
    textures : tuple of 2d-arrays, optional
        Data about the color of each pixels for each pixels.
        The function used here only works with grayscale pictures.
        The default is None.
    negatives : typles of bools, optional
        For each texture, precise if it should be applied on all of the image 
        except edges instead of edges (as a background for example). 
        The default is None.

    Returns
    -------
    2d-array
        Image with textures applied.

    """
    N, M = I.shape[:2]
    if textures is None:
        pencil = imageio.v2.imread(r'..\Img\Textures\pencil2.jpg', mode='F')
        paper = imageio.v2.imread(r'..\Img\Textures\paper.jpg', mode='F')/3 + 2*255/3
        marker = imageio.v2.imread(r'..\Img\Textures\feutre noir.jpg', mode='F')
        textures = (pencil, marker, paper)
        negatives = (False, False, True)
        sigmas = (4, .7, .7)
        ps = (65, 25, 25)
    N_text = len(textures)
    for texture in textures:
        assert texture.ndim == 2
    if sigmas is None:
        sigmas = (1.2,)*N_text
    if ps is None:
        ps = (35,)*N_text
    if negatives is None:
        negatives = (False,)*N_text
    assert len(sigmas) == len(ps) == len(textures) == len(negatives)
    textured = []
    for sigma, p, texture, negative in zip(sigmas, ps, textures, negatives):
        while texture.shape[0] < N or texture.shape[1] < M:
            texture = handle_borders(texture)
        texture = texture[:N, :M, ...]
        phi = np.inf if negative else -np.inf
        mask = xdog(I, sigma, p=p, phi=phi)
        textured.append(255 - (255-texture)*(mask/255))
    output = np.sum(textured, axis=0)
    return normalise(output, I_min_max=((N_text-1)*255, N_text*255))


def sharpening(I, sigma_blur=.1, sigma=2, p=25, epsilon=0, phi=.002):
    """
    Create a sharpening effect by highlighting edges.

    Parameters
    ----------
    I : 2d-array or 3d-array
        Data about the color of each pixel in a picture.
    sigma_blur : float, optional
        Standard deviation used to blur the background image. The default is .1.
    sigma : TYPE, optional
        DESCRIPTION. The default is 2.
    p : TYPE, optional
        DESCRIPTION. The default is 25.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 0.
    phi : TYPE, optional
        DESCRIPTION. The default is .002.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    mask = xdog(I, sigma, p, phi=np.inf)
    N, M = mask.shape
    sharpened = np.zeros_like(I).astype(np.float64)
    blurred = np.zeros_like(I).astype(np.float64)
    for i in range(3):
        blurred[:,:,i] = gaussian_filter(I[:,:,i], sigma_blur)
    blurred = normalise(blurred)
    for k in range(3):
        sharpened[:,:,k] = np.where(mask == 0, 0, blurred[:,:,k])
    output = np.where(sharpened >= epsilon, 1+np.tanh(phi*(sharpened+1e-10)), 1)
    output = normalise(output)
    return output
        

def teaser(I):
    list_data = ["I", "xdog(I, .9, p=20, epsilon=20, phi=.01, base='gray', n_tones=None)",
                 "xdog(I, .9, p=20, epsilon=20, phi=.01, base='gray', n_tones=3)",
                 "xdog(I, .9, p=20, epsilon=20, phi=.01, base='RGB', n_tones=None)",
                 "xdog(I, .9, p=20, epsilon=20, phi=.01, base='RGB', n_tones=5)",
                 "xdog(I, .9, p=20, epsilon=20, phi=.01, base='HSV', n_tones=None)",
                 "xdog(I, .9, p=20, epsilon=20, phi=.01, base='HSV', n_tones=3)",
                 "fdog(I, .2, 1.5, 1.3, 1, p=25, epsilon=40, phi=.01, n_tones=None, base='gray')",
                 "fdog(I, .2, 1.5, 1.3, 1, p=25, epsilon=40, phi=.01, n_tones=3, base='gray')",
                 "fdog(I, .2, 1.5, 1.3, 1, p=25, epsilon=40, phi=.01, n_tones=None, base='HSV')",
                 "fdog(I, .2, 1.5, 1.3, 1, p=25, epsilon=40, phi=.01, n_tones=3, base='HSV')",
                 "speed_lines(I, sigma_v=4, sigma=.5)",
                 "textures(I)",
                 "sharpening(I, sigma_blur=.1, sigma=.9, p=25, epsilon=0, phi=.001)",
                 ]
    list_titles = ['input', 'XDoG (grayscale)', 'XDoG (grayscale, 3 tones)',
                   'XDoG (RGB)', 'XDoG (RGB, 5 tones)', 'XDoG (HSV)', 'XDoG (HSV, 3 tones)',
                   'FDoG (grayscale)', 'FDoG (grayscale, 3 tones)', 'FDoG (HSV)', 'FDoG (HSV, 3 tones)', 
                   'speed lines', 'textures', 'sharpening']
    plt.axis('off')
    for i in tqdm(range(1, 15)):
        plt.subplot(3, 5, i)
        try:
            plt.imshow(eval(list_data[i-1]), cmap='gray')
            plt.title(list_titles[i-1])
        except IndexError:
            break
    plt.suptitle('Teaser')
    plt.show()
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from PIL import Image
    import imageio

    def resize_img(img, max_size=200):
        I = np.array(img).astype(np.float64)
        N, M = I.shape[:2]
        r = max(N,M) / max_size
        out = np.array(img.resize((int(M/r), int(N/r)), Image.Resampling.LANCZOS)).astype(np.uint8)
        return out

    big_pirate = imageio.v2.imread(r'..\Img\pirate.png')
    pirate = resize_img(Image.open(r'..\Img\pirate.png'))
    teaser(pirate)