from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from imageio import imread
from skimage.color import rgb2gray
import numpy as np

GRAYSCALE = 1
MIN_RES = 16


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Construct a Gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0,1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that represents a squared filter) to be used
                        in constructing the pyramid filter
    :return: tuple(pyr, filter_vec) where:
             pyr: the  resulting  pyramid as  a  standard  python  array with maximum length of max_levels,
                  where each element of the array is a grayscale image.
             filter_vec: normalized row vector of shape(1, filter_size) used for the pyramid construction
    """
    pyr = [im]
    filter_vec = create_filter(filter_size)
    i = 1  # counter to keep track of levels
    rows, cols = im.shape
    while i < max_levels and min(rows, cols) > MIN_RES:
        pyr.append(reduce(pyr[i - 1], filter_vec))
        rows, cols = pyr[i].shape
        i += 1
    return pyr, filter_vec


def create_filter(filter_size):
    """
    Creates a Gaussian filter of size filter_size
    :param filter_size: the size of the Gaussian filter
    :return: normalized row vector of shape(1, filter_size)
    """
    if filter_size == 1:
        return np.array([[1]])
    base_vec = np.convolve([1, 1], [1, 1])
    filter_vec = base_vec
    levels = (filter_size - 1) // 2  # this is the number of levels for the convolution
    for i in range(levels - 1):
        filter_vec = np.convolve(filter_vec, base_vec)
    filter_vec = filter_vec / 2 ** (2 * levels)  # normalize
    return filter_vec.reshape(1, filter_size)


def reduce(im, filter_vec):
    """
    Blurs and subsample an image to reduce it by half
    :param im: a grayscale image with double values in [0,1]
    :param filter_vec: normalized row vector of shape(1, filter_size) to blur with
    :return: reduced image
    """
    blurred_im = convolve(im, filter_vec)  # conv as row vector
    blurred_im = convolve(blurred_im, filter_vec.T)  # conv as col vector
    return blurred_im[::2, ::2]


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2)
    :return: the image represented by a matrix of type np.float64
    """
    im = imread(filename)
    if representation == GRAYSCALE and im.ndim == 3:
        im = rgb2gray(im)
        return im
    im = im.astype(np.float64)
    im /= 255
    return im
