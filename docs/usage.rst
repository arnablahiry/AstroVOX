Usage
=====

Launching the application
--------------------------

.. code-block:: bash

   astrovox

To launch directly into a cube instead of starting on the drop-zone landing
page, pass its path:

.. code-block:: bash

   astrovox <path_to_cube>

Both forms are also available as a module:

.. code-block:: bash

   python -m astrovox

Loading a cube
---------------

Open a FITS, HDF5, or NumPy cube from the drag-and-drop landing page or the
file picker.

- **FITS/HDF5** — cube info fields are pre-filled from the file's own
  metadata.
- **NumPy array** — fill in the cube type (PPP/PPV/PVP/VPP), field of view,
  spectral resolution, and quantity units so the render is on a physically
  meaningful scale.

Any cube axis ordering is supported and labelled accordingly:

- **PPP** — position–position–position, e.g. a spatial density field from a
  simulation snapshot.
- **PPV / PVP / VPP** — position–position–velocity cubes (and axis
  permutations thereof), the standard layout for spectral-line radio/mm-wave
  and IFU data.

Working with the render
-------------------------

Use the side panel to adjust the transfer function (Vmin/Vmax, Linear/Log/
Power scaling, colormap), the camera (free rotation or snap to an axis
plane), and visual aesthetics, before exporting a video or still frame. See
:doc:`gui` for a full walkthrough of every control.
