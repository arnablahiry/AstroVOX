AstroVOX
========

.. image:: ../assets/astrovox_banner.png
   :width: 100%
   :alt: AstroVOX logo

**AstroVOX** is a desktop visualization application for volumetric data cubes
commonly encountered in astronomy and astrophysics: interferometric and
single-dish spectral-line cubes (HI, CO), integral-field spectroscopy data,
and simulated cosmological/hydrodynamical volumes (e.g. CAMELS, IllustrisTNG).
It renders the cube as an interactive, GPU-accelerated 3D volume built on
`PyVista <https://pyvista.org/>`_ and `VTK <https://vtk.org/>`_.

.. grid:: 2

   .. grid-item-card:: Getting Started
      :link: installation
      :link-type: doc

      Install AstroVOX and load your first cube.

   .. grid-item-card:: Usage
      :link: usage
      :link-type: doc

      Launch the app, load a FITS/HDF5/NumPy cube, and drive the transfer
      function, camera, and export controls.

.. grid:: 2

   .. grid-item-card:: Graphical Interface
      :link: gui
      :link-type: doc

      Cube metadata, transfer function, camera and orientation, visual
      aesthetics, animation, export, and the 2D Projection / Moment 0 window.

   .. grid-item-card:: API Reference
      :link: autoapi/index
      :link-type: doc

      Auto-generated reference for every public class and function.

.. toctree::
   :maxdepth: 2
   :hidden:

   installation
   usage
   gui
   autoapi/index
