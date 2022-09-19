from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numbers import Number


def read_region(wsi, location, level, patch_size, custom_downsample):
    """
    Wrapper for wsi.read_region that handles downsampling. If we want a patch at 20x mag, but the WSI object only has 40x mag available then we read in a patch at 40x that is twice the desired size then downsample it.

    Parameters
    ----------
    wsi: openslide.OpenSlide
        The WSI object.

    location: tuple of ints
        The coordinates to read the patch in at the specified level before possibly downsampling.

    level: int
        The level at which to read the patch in from the WSI object.

    patch_size: int, tuple of ints
        The patch size to read in before possibliy downsampling.

    custom_downsample: int
        Downsample factor.

    Output
    ------
    img: PIL.Image.Image
        The region image.
    """

    # process read patch size to tuple
    if isinstance(patch_size, Number):
        _read_patch_size = (int(patch_size), int(patch_size))
    else:
        _read_patch_size = tuple(patch_size)
    assert len(_read_patch_size) == 2

    # read in raw patch from image
    img = wsi.read_region(location=location,
                          level=level,
                          size=_read_patch_size
                          ).convert('RGB')

    # possible resize image
    if custom_downsample > 1:
        target_patch_size = (_read_patch_size[0] // custom_downsample,
                             _read_patch_size[1] // custom_downsample)

        img = img.resize(size=target_patch_size)

    return img


def make_image_grid(imgs, pad=3, n_cols=10):
    """
    Makes a grid of PIL images.

    Parameters
    ----------
    imgs: list of PIL imags.
        The images to display in a grdie.

    pad: int
        Amount of padded to put between images.

    n_cols: int
        Maximum number of columns to put in the grid.

    Output
    ------
    grid: PIL.Image
        The image grid.
    """
    n_images = len(imgs)
    n_rows = int(np.ceil(n_images / n_cols))

    img_height = max([img.size[0] for img in imgs])
    img_width = max([img.size[1] for img in imgs])

    grid = Image.new('RGBA',
                     size=(n_rows * img_height + (n_rows - 1) * pad,
                           n_cols * img_width + (n_cols - 1) * pad),
                     color=(255, 255, 255))

    for idx in range(n_images):

        row_idx = idx // n_cols
        col_idx = idx % n_cols

        grid.paste(imgs[idx],
                   box=(row_idx * (img_height + pad),
                        col_idx * (img_width + pad))
                   )
    return grid


def save_fig(fpath, dpi=100, bbox_inches='tight'):
    """
    Saves and closes a plot.
    """
    plt.savefig(fpath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()


def get_idxs_largest(values, k):
    """
    Gets the indices of the k largest elements of an array.

    Parameters
    ----------
    values: array-like
        The values.

    Output
    ------
    idxs: array-like, shape (k, )
        The idxs of the largest elements.
    """
    values = np.array(values)
    assert values.ndim == 1
    return np.argpartition(values, -k)[-k:]


def get_idxs_smallest(values, k):
    """
    Gets the indices of the k smallest elements of an array.

    Parameters
    ----------
    values: array-like
        The values.

    Output
    ------
    idxs: array-like, shape (k, )
        The idxs of the smallest elements.
    """
    values = np.array(values)
    assert values.ndim == 1
    return np.argpartition(values, k)[:k]
