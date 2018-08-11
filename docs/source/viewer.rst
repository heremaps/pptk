Point cloud viewer
==================

Controls
--------
**View manipulation**. The camera look at position is denoted by a
red-green-blue cursor, with segments corresponding to x, y and z axes. Use the
left mouse button to rotate. Hold down shift to pan. Double-clicking moves the
look at position to a point near the mouse cursor.

**Point selection**. Hold down ctrl while clicking or dragging a box to add 
points to a selection. Hold down ctrl+shift to remove from a selection. Use the
right mouse button to clear an existing selection.

**Hot keys**.

===  ========================================================
key	 Description
===  ========================================================
5	   Toggle between orthographic/perspective projection
1	   Look along +y direction
3	   Look along -x direction
7	   Look along -z direction
c	   Set look at position to mean position of selected points
[	   Toggle previous attribute set
]	   Toggle next attribute set
===  ========================================================

Methods
-------

.. autoclass:: pptk.viewer
   :members:
   :special-members:
