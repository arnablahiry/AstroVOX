Graphical Interface
====================

AstroVOX is a single window: a square, interactive 3D viewport on the left
and a controls column on the right. The controls column is organized into
the sections below, top to bottom.

.. contents:: On this page
   :local:
   :depth: 2

Cube metadata
-------------

Name, telescope/simulation origin, cube type (PPP/PPV/PVP/VPP), field of
view with physical unit, spectral resolution with unit, and quantity name
and unit — pre-filled from the file's own header/attrs for FITS/HDF5, editable
for any cube.

Transfer function
------------------

Independent V\ :sub:`min`/V\ :sub:`max` range sliders, and Linear, Log, or
Power (with an adjustable gamma) scaling of the volume's color and opacity
mapping, plus a colormap selector.

Camera and orientation
------------------------

Free rotation via mouse/trackpad, or snap the view to a fixed X-Y, Y-Z, or
X-Z plane.

Visual aesthetics
-------------------

Toggles for mini orientation axes, main axes labels, tick marks and labels,
grid lines, colorbar, scale bar, and cube outline (with selectable line
style and thickness).

Animation
---------

Independent azimuth (horizontal) and elevation (vertical) auto-rotation,
each with its own play/stop control and adjustable angular velocity.

Export
------

Record a rotating fly-around to video at a chosen frame rate, or save the
current view as a static frame in a chosen image format. A blinking on-frame
indicator and elapsed timer show while a recording is in progress.

2D Projection / Moment 0
--------------------------

Computes and displays a 2D projection in a separate window:

- For a **PPP** cube, a projection along the volume's *current* line of
  sight (the camera orientation at the moment the projection is computed).
- For a **PPV/PVP/VPP** cube, a real moment-0 map — the cube summed along
  its spectral axis and scaled by the spectral resolution.

The window has its own controls (independent of the main viewport):
spatial resolution and projected quantity unit, V\ :sub:`min`/V\ :sub:`max`,
scale, colormap, interpolation, plot theme, zoom, and export. A FITS PPV
cube with a valid WCS renders real sky-coordinate axes (RA/Dec) via
`astropy.visualization.wcsaxes`.

Light/dark theme
------------------

A theme toggle restyles the whole interface, and a one-click reset returns
to the drop-zone landing page.
