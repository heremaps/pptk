.. title:: Visualizing the Tanks and Temples dataset

Tanks and Temples
=================

`Tank and Temples <https://www.tanksandtemples.org>`_
is a benchmark that uses Lidar point clouds as ground truth
for benchmarking the quality of image-based 3-d reconstruction algorithms.
The point clouds are stored as .ply files.
Here we show how to load and visualize these point clouds.

Download :code:`Truck.ply` (392 MB)
`[link] <https://docs.google.com/uc?export=download&id=0B-ePgl6HF260NlB1MXF1ZUs0c0U>`__.

Import required Python packages.
Here we use the plyfile Python package to read .ply files.

.. code-block:: python

    >>> import pptk
    >>> import numpy as np
    >>> import plyfile

.. note::
   If :code:`pip install plyfile` does not work,
   simply save a local copy of plyfile.py from plyfile's `github page <https://github.com/dranjan/python-plyfile>`__.

Read vertices in :code:`Truck.ply`.

.. code-block:: python

    >>> data = plyfile.PlyData.read('Truck.ply')['vertex']

Use per-vertex attributes to make numpy arrays :code:`xyz`, :code:`rgb`, and :code:`n`.

    >>> xyz = np.c_[data['x'], data['y'], data['z']]
    >>> rgb = np.c_[data['red'], data['green'], data['blue']]
    >>> n = np.c_[data['nx'], data['ny'], data['nz']]

Visualize.

    >>> v = pptk.viewer(xyz)
    >>> v.attributes(rgb / 255., 0.5 * (1 + n))

Use :kbd:`[` and :kbd:`]` to toggle between attributes.

.. |truck_rgb| image:: images/tanks_and_temples_truck_rgb.jpg
   :width: 340px
   :align: middle

.. |truck_n| image:: images/tanks_and_temples_truck_n.jpg
   :width: 340px
   :align: middle

.. rst-class:: image-grid

.. table::
   :widths: 350 350
   :align: center

   =========== =========
   |truck_rgb| |truck_n|
   =========== =========

.. rst-class:: caption

   +----------------------------------------------------------------------------------------------+
   | The :file:`Truck.py` point cloud from `Tanks and Temples <https://www.tanksandtemples.org>`_ |
   | visualized using :py:meth:`pptk.viewer`. Points are colored by RGB (left) and normal (right) |
   +----------------------------------------------------------------------------------------------+

The above procedure can be repeated for the other point clouds in the dataset.
