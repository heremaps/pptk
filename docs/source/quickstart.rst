Getting started
===============

Install pptk.

    >>> pip install pptk

.. note::
   pptk requires 64-bit Python and is only :code:`pip install`-able on versions of Python
   that have a corresponding pptk wheel file on `PyPI <https://pypi.org/project/pptk/>`__.

In Python, generate 100 random 3-d points, and 

.. code-block:: python

    >>> import numpy as np
    >>> x = np.random.rand(100, 3)

Visualize.

.. code-block:: python

    >>> import pptk
    >>> v = pptk.viewer(x)

Set point size to 0.01.

.. code-block:: python

    >>> v.set(point_size=0.01)
