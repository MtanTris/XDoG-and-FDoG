import numpy as np
from utils import apply_to_components, regular_segmentation, normalise
from scipy.ndimage import gaussian_filter

k = 1.6

def xdog(I, sigma, p=20, epsilon=20, phi=.01, base='gray', n_tones=None):
    """
    Apply the XDoG algorithm to a picture I.

    Parameters:
        I: array of dimension 2 or 3
            Picture on which XDoG will be applied.
        sigma: float
            Standard deviation of the Gaussians used to convolve the image.
            Depitcs the spatial support of XDoG. Use higher values to make the output coarser and more 
            abstract.
            We often put its value around 2, but the value depends on the size of the picture.
        p: float
            Depicts the sharpness of the output. Use higher values to catch more subtle details, at the
            risk of making the algorithm more sensitive to noise.
            We often use values of p between 5 and 30.
        epsilon: float
            Describes at which value the filter goes from a constant to a hyperbolic
            tangent. Use a higher value to make the output darker and more dramatic.
            We often use values of epsilon between 15 and 65.
        phi: float or np.inf
            Describe the slope of the second part of the filter.
            Use a lower value to add shades of gray to the result. Use a negative value to reverse the
            colors of the output.
            We often use values between 0.008 and 0.02 as the output doesn't change much past 0.02.
        base: str ('gray', 'RGB' or 'HSV')
            Tells in which color base the XDoG algorithm will be applied.
                - If base is set to 'gray', then the algorithm will run once on the image, preemptively
                turned to a grayscale image.
                - If base is set to 'RGB', then the algorithm will run on every component (red, green and
                blue) of the image. The output will be a combination of the three results.
                - If base is set to 'HSV', then the algorithm will run only on the value component. The
                result will be combined with the original hue and saturation components to make up the
                output.
        n_tones: int or None
            Number of tones of the output. Can be used as an additional effect. Set this parameter to None
            to ignore this step of the algorithm and to make the output keep all of its colors.
            When it is not set to None, we often restrict the output to 3 or 4 tones.
    
    Returns: 
        Output of the XDoG algorithm applied to the given color base.
    """
    return apply_to_components(I, xdog_1component, base, n_tones, sigma, p, epsilon, phi)


def xdog_1component(I, sigma, p, epsilon, phi, n_tones=None):
    #Convolve with both standard deviations.
    g1 = gaussian_filter(I, sigma)
    gk = gaussian_filter(I, k*sigma)

    #Rescale phi and epsilon with respect to p.
    phi0 = phi*(p+1)
    epsilon0 = epsilon/(p+1)
    tau = p/(p+1)

    #Compute the difference of gaussians and apply the filter
    D = g1 - tau*gk
    filtered = np.where(D >= epsilon0, 1+np.tanh(phi0*(D)), 1)

    #Eventually quantify the tones of the output.
    if n_tones is not None:
        filtered = regular_segmentation(filtered, n_tones, vmin=1, vmax=2)
    out = 255*normalise(filtered)
    return out