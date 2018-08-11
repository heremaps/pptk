Getting started
===============

Install pptk.

    >>> pip install pptk

In Python, generate 100 random 3-d points, and visualize it using pptk's viewer.

    >>> import pptk
    >>> import numpy as np
    >>> x = np.random.rand(100, 3)
    >>> pptk.viewer(x)

Set the point size to 0.01

    >>> v.set(point_size=0.01)
