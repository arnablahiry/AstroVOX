Installation
============

Requirements
------------

- Python ≥ 3.9
- PyVista and VTK — 3D volume rendering
- Astropy — FITS I/O and WCS handling
- NumPy, SciPy
- imageio / imageio-ffmpeg — video export

From PyPI
---------

.. code-block:: bash

   pip install astrovox

From source
-----------

.. code-block:: bash

   git clone https://github.com/arnablahiry/AstroVOX.git
   cd AstroVOX
   pip install -e .

Supported input formats
------------------------

- **FITS** (``.fits``, ``.fit``, ``.fts``) and **HDF5** (``.h5``, ``.hdf5``) —
  metadata (header/attrs, WCS, intensity units) is read directly from the
  file and used to pre-fill the cube info fields.
- **NumPy arrays** (``.npy``, ``.npz``) — carry no physical metadata, so the
  cube type, field of view, spectral resolution, and quantity units are
  filled in manually for a physically meaningful render.
