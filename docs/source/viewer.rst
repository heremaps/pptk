Point cloud viewer
==================

The :py:meth:`pptk.viewer` function enables one to directly visualize large point clouds in Python.
It accepts as input any Python variable that can be cast as a 3-column numpy array (i.e. via :py:func:`np.asarray`).
It interprets the columns of such input as the x, y, and z coordinates of a point cloud.

The viewer itself runs as a standalone operating system process separate from Python.
The user can query and manipulate the viewer via the handle that is returned by :py:meth:`pptk.viewer`.
The handle encapsulates the details of communicating with the viewer (e.g. the viewer process's port number)
and provides methods for querying and manipulating the viewer.

The viewer supports interactive visualization of tens of millions of points via an octree-based level-of-detail renderer.
At startup the viewer organizes the input points into an octree.
As the viewpoint is being manipulated,
the octree is used to approximate groups of far away points as single points and cull points that are outside the view frustum,
thus greatly reducing the number of points being rendered.
Once there are no more changes to the viewpoint,
the viewer then proceeds to perform a more time consuming detailed rendering of the points.

.. note::
  Currently the viewer crashes for larger number of points.
  The actual number depends on system and GPU memory.
  On certain machines this is known to start happening around 100M points.

Controls
--------

**View manipulation**.
The camera look at position is denoted by a red-green-blue cursor,
with segments corresponding to x, y and z axes.
Perform :kbd:`LMB` drag to rotate the viewpoint in a turntable fashion.
Perform :kbd:`LMB` drag while hold :kbd:`Shift` to pan the viewpoint.
Double clicking the :kbd:`LMB` moves the look at position to a point near the mouse cursor.

.. note::
   :kbd:`LMB`/:kbd:`RMB` stands for left/right mouse button

**Point selection**.
Hold :kbd:`Ctrl`/:kbd:`Ctrl-Shift` while :kbd:`LMB` dragging a box to add/remove points to the current selection.
Hold :kbd:`Ctrl`/:kbd:`Ctrl-Shift` while :kbd:`LMB` clicking on a point
to add/remove a *single* point from the current selection.
Click the :kbd:`RMB` to clear the current selection.
Use :py:meth:`pptk.viewer.get` to query the selected point indices.

.. note::
   If you are using a Mac, use :kbd:`⌘` in place of :kbd:`Ctrl`.

**Hot keys**.

========  ========================================================
key	      Description
========  ========================================================
:kbd:`5`	Toggle between orthographic/perspective projection
:kbd:`1`	Look along +y direction
:kbd:`3`	Look along -x direction
:kbd:`7`	Look along -z direction
:kbd:`C`	Set look at position to mean position of selected points
:kbd:`[`	Toggle previous attribute set
:kbd:`]`	Toggle next attribute set
========  ========================================================

Methods
-------

.. autoclass:: pptk.viewer
   :members:
   :special-members:
