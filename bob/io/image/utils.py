import numpy as np


def to_matplotlib(img):
    '''Returns a view of the image compatible with matplotlib.

    Parameters
    ----------
    img : numpy.ndarray
        A 2 or 3 dimensional array containing an image in bob style: For a 2D
        array (grayscale image) should be ``(y, x)``; For a 3D array (color
        image) should be ``(n, y, x)``.

    Returns
    -------
    numpy.ndarray
        A view of the ``img`` compatible with
        :py:func:`matplotlib.pyplot.imshow`.
    '''
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    return img


def to_bob(img):
    '''Returns a view of the image compatible with Bob.

    Parameters
    ----------
    img : numpy.ndarray
        A 2 or 3 dimensional array containing an image in matplotlib style: For
        a 2D array (grayscale image) should be ``(y, x)``; For a 3D array
        (color image) should be ``(y, x, n)``.

    Returns
    -------
    numpy.ndarray
        A view of the ``img`` compatible with Bob ``(n, y, x)`` for 3D images.
    '''
    if img.ndim == 3:
        img = np.moveaxis(img, -1, 0)
    return img


def imshow(img, cmap=None, **kwargs):
    '''Plots the images that are returned by :py:func:`bob.io.base.load`

    Parameters
    ----------
    img : numpy.ndarray
        A 2 or 3 dimensional array containing an image in
        bob style: For a 2D array (grayscale image) should be ``(y, x)``;
        A 3D array (color image) should be in the ``(n, y, x)`` format.
    cmap : matplotlib.colors.Colormap
        Colormap, optional, default: ``None``.
        If ``cmap`` is ``None`` and ``img.ndim`` is 2, defaults to 'gray'.
        ``cmap`` is ignored when ``img`` has RGB(A) information.
    **kwargs
        These are passed directly to :py:func:`matplotlib.pyplot.imshow`

    Returns
    -------
    object
        Returns whatever ``plt.imshow`` returns.
    '''
    import matplotlib.pyplot as plt

    if cmap is None and img.ndim == 2:
        cmap = 'gray'

    return plt.imshow(to_matplotlib(img), cmap=cmap, **kwargs)
