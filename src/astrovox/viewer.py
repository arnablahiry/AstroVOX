from __future__ import annotations

from pathlib import Path
from typing import Optional

import math

import numpy as np
import pyvista as pv
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


_MATHTEXT_CACHE: dict = {}
_MATHTEXT_DPI = 200


def _render_mathtext_rgba(text: str, color=(1.0, 1.0, 1.0), fontsize: int = 18):
    """Rasterize `text` (matplotlib mathtext — a LaTeX-like subset:
    subscripts/superscripts, Greek letters, \\odot, fractions, etc., not
    a full LaTeX engine) to an (H, W, 4) uint8 RGBA array on a
    transparent background, for display where a plain VTK text actor
    can't render math (the colorbar title, and a numpy cube's axis
    titles once a physical unit has been supplied — see
    KinematicVolumeViewer.set_manual_axis_scale). Text outside any
    ``$...$`` span renders as plain (bold) text; matplotlib's mathtext
    parser handles that mixed mode natively. Results are cached since
    rendering is comparatively expensive and the same label text is
    often re-requested (e.g. on every theme toggle)."""
    key = (text, tuple(color), fontsize)
    cached = _MATHTEXT_CACHE.get(key)
    if cached is not None:
        return cached

    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    def _render(s):
        fig = plt.figure()
        fig.patch.set_alpha(0.0)
        # "fontfamily" only covers plain text — the math (``$...$``) spans
        # are typeset by mathtext's own font selection, controlled
        # separately via this rcParam; "dejavuserif" is the serif
        # counterpart of the plain-text default so both halves of mixed
        # text match.
        with matplotlib.rc_context({"mathtext.fontset": "dejavuserif"}):
            fig.text(0.5, 0.5, s, color=color, fontsize=fontsize, fontweight="bold", fontfamily="serif", ha="center", va="center")
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png", dpi=_MATHTEXT_DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
            finally:
                plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGBA"))

    try:
        rgba = _render(text)
    except (ValueError, RuntimeError):
        # Malformed mathtext (e.g. an unmatched/empty "$...$" span from a
        # half-typed unit) must never crash the app — fall back to the
        # literal text with "$" escaped so it renders as plain characters.
        rgba = _render(text.replace("$", r"\$"))
    _MATHTEXT_CACHE[key] = rgba
    return rgba


def _nice_ticks(vmin: float, vmax: float, target: int = 8) -> list[float]:
    """A "nice number" tick locator (1-2-2.5-5-10 stepping, matching the
    kind matplotlib/D3 use) — searches candidate round step sizes across
    magnitudes and picks whichever yields a tick count closest to
    `target`, spanning [vmin, vmax]. Used by the custom RA/Dec tick-label
    overlay (see KinematicVolumeViewer._rebuild_custom_tick_labels) to
    pick its own zoom-adaptive tick density."""
    span = vmax - vmin
    if span <= 0:
        return [vmin]
    best = None
    for exponent in range(-6, 7):
        mag = 10.0 ** exponent
        for base in (1, 2, 2.5, 5):
            step = base * mag
            start = math.ceil(vmin / step - 1e-9) * step
            count = int(math.floor((vmax - start) / step + 1e-9)) + 1
            if count < 2:
                continue
            score = abs(count - target)
            if best is None or score < best[0] or (score == best[0] and count < best[3]):
                best = (score, step, start, count)
    _, step, start, count = best
    return [round(start + i * step, 10) for i in range(count)]


def _format_ticks_with_abbreviation(components, units) -> list[str]:
    """Given a list of (sign, major, minor, seconds) tuples in ascending
    tick order, render each as sexagesimal text — but the way astropy's
    own WCSAxes ticks do it: only the *first* tick (and any tick whose
    leading components actually change from the previous one) spells out
    the full "22h16m09.5s"; a neighbour that only differs in the trailing
    component shows just that part, e.g. "09.1s"."""
    u1, u2, u3 = units
    show_sign = u1 == "°"  # only Dec (degrees) carries an explicit sign
    strings = []
    prev = None
    for sign, a, b, c in components:
        if prev is not None and sign == prev[0] and a == prev[1] and b == prev[2]:
            text = f"{c:04.1f}{u3}"
        elif prev is not None and sign == prev[0] and a == prev[1]:
            text = f"{b:02d}{u2}{c:04.1f}{u3}"
        else:
            sign_str = ("-" if sign < 0 else "+") if show_sign else ""
            text = f"{sign_str}{a:02d}{u1}{b:02d}{u2}{c:04.1f}{u3}"
        strings.append(text)
        prev = (sign, a, b, c)
    return strings


class KinematicVolumeViewer:
    """Standalone 3D viewer for kinematic spectral cubes with a light/dark theme toggle."""

    def __init__(
        self,
        cube,
        vel_scale: Optional[float] = None,
        opacity="sigmoid",
        cmap="plasma",
        plotter: Optional[pv.Plotter] = None,
        show_moment0: bool = False,
        embed_controls: bool = True,
        axis_labels: Optional[tuple] = None,
        spatial_scale: Optional[tuple] = None,
        colorbar_title: str = "Intensity",
        axis_ranges: Optional[tuple] = None,
        axis_label_formats: Optional[tuple] = None,
        axis_tick_formatters: Optional[tuple] = None,
        axis_tick_units: Optional[tuple] = None,
    ):
        self.opacity = opacity
        self.cmap = cmap
        self.show_moment0 = show_moment0
        # When False, no theme checkbox, clim sliders, or scalar bar are
        # drawn on top of the render window itself — an external GUI (e.g.
        # a Qt side panel) drives set_clim()/set_theme() instead.
        self.embed_controls = embed_controls
        # (xlabel, ylabel, zlabel) for the orientation-widget axes, e.g.
        # ("RA", "Dec", "km/s") for an observed cube or ("kpc", "kpc",
        # "km/s") for a simulated one.
        self.axis_labels = axis_labels or ("X", "Y", "Z")
        # (value_per_voxel, unit) used to convert the fixed-length scale
        # bar's on-screen size into a physical length, e.g. (0.025,
        # "arcsec"); None means the scale is unknown, so the bar shows a
        # static "px" label instead of a dynamically-updating length.
        self.spatial_scale = spatial_scale
        # Physical quantity of the colour axis (e.g. the FITS BUNIT header
        # value, "Jy/beam") — shown as the colorbar's title.
        self.colorbar_title = colorbar_title
        # ((x_min, x_max), (y_min, y_max), (z_min, z_max)) real-world
        # values (e.g. arcsec offset / km/s, or kpc) to display at the
        # cube's geometric bounds — None per-axis falls back to raw voxel
        # indices. A mutable list (not the tuple callers pass in) so a
        # numpy-array cube's axes can be given real units interactively
        # later via set_manual_axis_scale, rather than only at load time.
        self.axis_ranges = list(axis_ranges) if axis_ranges is not None else [None, None, None]
        self.axis_label_formats = axis_label_formats or ("%.2f", "%.2f", "%.1f")
        # (x_formatter, y_formatter, z_formatter) — per-axis callables
        # ``value -> str`` (e.g. real sexagesimal RA/Dec text) used by our
        # *own* custom tick-label overlay instead of vtkCubeAxesActor's
        # built-in numeric labels; None per axis means that axis just uses
        # vtkCubeAxesActor's native labels/axis_label_formats as normal.
        # A custom overlay is needed because vtkCubeAxesActor's own
        # SetAxisLabels(axis, strings) silently discards any array whose
        # length doesn't match the tick count it computes internally
        # (confirmed empirically — there's no public API to read that
        # count back, or to control it), which rules out ever showing
        # fewer/more ticks than whatever it happens to pick.
        self.axis_tick_formatters = list(axis_tick_formatters) if axis_tick_formatters is not None else [None, None, None]
        # Per-axis (major_symbol, minor_symbol, seconds_symbol), e.g.
        # ("h", "m", "s") or ("°", "'", '"') — used both to render each
        # tick's text and to decide (via the "°" check) whether that axis
        # shows an explicit +/- sign.
        self.axis_tick_units = list(axis_tick_units) if axis_tick_units is not None else [None, None, None]
        # Per-axis raw unit text (e.g. "$M_\\odot/h$") supplied
        # interactively for a numpy-array cube once Field of view/
        # Spectral Resolution + a unit are both given (see
        # set_manual_axis_scale) — shown, LaTeX-rendered, in parentheses
        # after the axis title ("X ($M_\\odot/h$)").
        self._manual_axis_units = [None, None, None]
        self._custom_tick_actors = []
        self._custom_tick_edge_dirs = []
        self._custom_tickmark_actors = []
        self._custom_gridline_actors = []
        self._custom_tick_state_key = None
        self._custom_title_actors = []
        self._custom_title_edge_dirs = []
        self._custom_title_state_key = None
        self._camera_ref_distance = None
        self._colorbar_actor = None
        self._colorbar_frame_actor = None
        self._colorbar_tick_actor = None
        self._colorbar_title_actor = None
        self._colorbar_value_label_actors = []
        self._scale_bar_actor2d = None
        self._scale_bar_text_actor = None
        self._scale_bar_frac = 0.14  # fixed fraction of viewport width
        self._label_font_size = 14  # shared by the scale bar and colorbar tick labels
        self.colorbar_visible = True
        self.scalebar_visible = True

        # Visual-aesthetics toggles (see set_mini_axes_visible etc.).
        self.mini_axes_visible = True
        self.show_main_axes_labels = False
        self.show_axis_ticks = False
        self.show_grid_lines = False
        self.cube_axes_actor = None
        cube = np.asarray(cube, dtype=np.float32)
        if cube.ndim == 2:
            cube = cube[np.newaxis, :, :]
        self.cube = cube

        nz, ny, nx = cube.shape
        if vel_scale is None:
            spatial_max = max(nx, ny)
            vel_scale = (spatial_max / nz) if nz < spatial_max else 1.0
        self.vel_scale = vel_scale

        self.grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(1.0, 1.0, self.vel_scale),
        )
        linear_values = np.transpose(cube, (2, 1, 0)).flatten(order="F")
        self.grid.point_data["intensity"] = linear_values

        self.d_min = float(np.nanmin(cube))
        self.d_max = float(np.nanmax(cube))

        # Log-scale colour/opacity mapping (see set_value_scale) needs a
        # positive floor — real data (astronomical intensity maps
        # especially) routinely has zero/negative noise-floor values,
        # which log10 can't represent. Values at or below the floor all
        # collapse to the same (darkest) end of the colour map, which is
        # the conventional way image-viewers handle this.
        positive = linear_values[np.isfinite(linear_values) & (linear_values > 0)]
        self._log_floor = float(positive.min()) if positive.size else 1e-10
        self.grid.point_data["intensity_log"] = np.log10(np.clip(linear_values, self._log_floor, None))
        self.value_scale = "linear"
        # Default Vmin a bit above the noise floor rather than at the raw
        # minimum, which on real data is usually just a random negative
        # noise dip — starting there buries the volume in noise on load.
        # Estimated robustly (median + k*MAD, the standard sigma-clip-style
        # noise estimate for radio/IR cubes) rather than a fixed percentile,
        # so it adapts to how noisy a given cube actually is.
        median = float(np.nanmedian(cube))
        mad = float(np.nanmedian(np.abs(cube - median)))
        noise_sigma = 1.4826 * mad
        default_vmin = min(median + 2 * noise_sigma, self.d_max)
        self.current_clim = [default_vmin, self.d_max]
        self.is_dark_theme = False

        if self.show_moment0:
            self.moment0 = np.nansum(cube, axis=0)
            self._moment0_clim = [
                float(np.nanpercentile(self.moment0, 1)),
                float(np.nanpercentile(self.moment0, 99.5)),
            ]
            self.moment0_actor = None

        self.plotter = plotter or pv.Plotter()
        self.cube_outline_visible = True
        self.cube_outline_thickness = 1
        self.cube_outline_style = "solid"
        self.bbox_actor = None
        self._rebuild_cube_outline()
        if self.spatial_scale is not None:
            self._build_scale_bar()
            self.plotter.renderer.GetActiveCamera().AddObserver(
                "ModifiedEvent", lambda obj, evt: self._update_scale_bar()
            )
        self._build_cube_axes()
        # Registered unconditionally (not just when axis_tick_formatters
        # is already populated) — a numpy cube starts with no formatters
        # set at all (the user hasn't typed a Field of view/Spectral
        # Resolution yet) and only gets them later via
        # set_manual_axis_scale(); gating registration on their presence
        # at construction time meant this observer was silently never
        # added for numpy cubes, so tick/title orientation would freeze
        # after being set once and never track further camera moves
        # (rotate, or the axis-snap pills) the way it does for FITS/HDF5
        # cubes, whose formatters are already known at construction.
        self.plotter.renderer.GetActiveCamera().AddObserver(
            "ModifiedEvent", lambda obj, evt: self._maybe_rebuild_custom_tick_labels()
        )
        self.apply_theme()

        if self.embed_controls:
            self.plotter.add_checkbox_button_widget(
                self.toggle_theme,
                value=self.is_dark_theme,
                color_on="gray",
                color_off="lightgray",
                position=(10, 10),
                size=30,
                border_size=2,
            )
            self.plotter.add_text("Toggle Theme", position=(50, 15), font_size=10, name="theme_label")

    def toggle_theme(self, state):
        self.is_dark_theme = bool(state)
        self.apply_theme()

    def set_theme(self, is_dark: bool):
        """Externally driven theme switch (e.g. from a host GUI's own toggle button)."""
        self.is_dark_theme = bool(is_dark)
        self.apply_theme()

    def set_clim(self, vmin: float, vmax: float):
        """Externally driven clim update (e.g. from a host GUI's own
        sliders). Rebuilds just the volume's small (256-entry) colour/
        opacity lookup table in place rather than the whole volume actor
        — a full rebuild re-uploads the entire 3D scalar texture to the
        GPU, expensive enough that dragging vmin/vmax while a rotation
        animation is running visibly stutters the rotation."""
        if vmin >= vmax:
            return
        self.current_clim = [vmin, vmax]
        self._update_clim_fast()

    def _value_range_for_clim(self, clim):
        if self.value_scale == "log":
            lo = max(clim[0], self._log_floor)
            hi = max(clim[1], self._log_floor)
            if hi <= lo:
                hi = lo * 10.0
            return (math.log10(lo), math.log10(hi))
        return tuple(clim)

    def _update_clim_fast(self):
        if getattr(self, "_volume_actor", None) is None:
            self.redraw_volume()
            return

        from pyvista.plotting.colors import get_cmap_safe
        from pyvista.plotting.tools import opacity_transfer_function
        from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction

        lo, hi = self._value_range_for_clim(self.current_clim)

        # Rebuilding the transfer functions fresh from the same canonical
        # colour/opacity source every time (rather than incrementally
        # rescaling whatever nodes are already there) avoids any risk of
        # drift compounding over many rapid updates — a real bug hit
        # during development, where repeatedly rescaling in place caused
        # the effective range to visibly wander after a couple hundred
        # slider ticks.
        n = 256
        colors = get_cmap_safe(self.cmap)(np.linspace(0.0, 1.0, n))
        alphas = opacity_transfer_function(self.opacity, n).astype(np.float64) / 255.0
        xs = np.linspace(lo, hi, n)

        ctf = vtkColorTransferFunction()
        sof = vtkPiecewiseFunction()
        for i in range(n):
            ctf.AddRGBPoint(xs[i], colors[i, 0], colors[i, 1], colors[i, 2])
            sof.AddPoint(xs[i], alphas[i])

        prop = self._volume_actor.prop
        prop.SetColor(0, ctf)
        prop.SetScalarOpacity(0, sof)

        if not self.show_moment0:
            if not self._update_colorbar_value_labels():
                self._rebuild_colorbar()

        self.plotter.render()

    def _update_colorbar_value_labels(self) -> bool:
        """Just refresh the colorbar's own displayed numbers, in place —
        used after a clim-only change. Unlike _rebuild_colorbar_trim,
        this never recreates actors or forces a synchronous VTK render
        pass (GetScalarBarRect there needs one) — at slider-drag
        frequency, that synchronous render is reentrant enough with
        Qt's own paint/layout cycle to visibly corrupt unrelated widget
        geometry (observed as overlapping Visual Aesthetics pills).
        Returns False (caller should fall back to a full rebuild) if the
        label actors don't exist yet."""
        labels = self._colorbar_value_label_actors
        if len(labels) != self._COLORBAR_N_LABELS:
            return False
        vmin, vmax = self.current_clim
        n = self._COLORBAR_N_LABELS
        for i, actor in enumerate(labels):
            frac = i / (n - 1) if n > 1 else 0.5
            if self.value_scale == "log":
                lo = max(vmin, self._log_floor)
                hi = max(vmax, self._log_floor)
                if hi <= lo:
                    hi = lo * 10.0
                log_lo, log_hi = math.log10(lo), math.log10(hi)
                value = 10 ** (log_lo + frac * (log_hi - log_lo))
            else:
                value = vmin + frac * (vmax - vmin)
            actor.SetInput(f"{value:.2e}")
        return True

    def set_opacity(self, opacity):
        """Externally driven opacity-transfer-function update (e.g. from a
        host GUI's own Linear/Log/Power selector). Accepts anything
        PyVista's ``add_volume(opacity=...)`` accepts: a preset name or an
        array of per-scalar opacity values."""
        self.opacity = opacity
        self.redraw_volume()

    def set_value_scale(self, mode: str):
        """Switch the volume's data-to-colour mapping (and, in turn, the
        colorbar) between "linear" and "log10" — distinct from the
        Linear/Log/Power *opacity* transfer function (set_opacity),
        which only reshapes alpha falloff and leaves colour mapping
        linear regardless. vmin/vmax (current_clim) stay in real,
        linear units always; only the mapping is log."""
        self.value_scale = mode
        self.redraw_volume()

    def set_cmap(self, cmap: str):
        """Externally driven colormap update (e.g. from a host GUI's own
        colormap dropdown)."""
        self.cmap = cmap
        self.redraw_volume()

    def apply_theme(self):
        if self.is_dark_theme:
            bg_color = "black"
            fg_color = "white"
        else:
            bg_color = "white"
            fg_color = "black"

        # Hold off on rendering (redraw_volume/_rebuild_axes etc. all pass
        # render=False) until every piece — background, volume, colorbar,
        # axes, scale bar — has been restyled, so the theme switches in
        # one atomic frame instead of visibly flipping piece by piece
        # across several frames.
        self.plotter.set_background(bg_color)
        if hasattr(self, "bbox_actor"):
            self.bbox_actor.prop.color = fg_color

        if self.embed_controls:
            self.plotter.add_text("Toggle Theme", position=(50, 15), font_size=10, color=fg_color, name="theme_label")

        self._rebuild_axes()
        self.redraw_volume(render=False)
        if self.show_moment0:
            self.redraw_moment0()
        if self.embed_controls:
            self.redraw_sliders(fg_color)
        if self.spatial_scale is not None:
            self._style_scale_bar()
            self._update_scale_bar()
        self._style_cube_axes()

        self.plotter.render()

    def _rebuild_axes(self):
        """(Re)build the corner orientation widget with the cube's actual
        axis labels (e.g. RA/Dec/km/s), pastel arrow colours in dark mode
        (so they don't glare against the black background), and a serif
        label font."""
        label_rgb = (1.0, 1.0, 1.0) if self.is_dark_theme else (0.0, 0.0, 0.0)
        if self.is_dark_theme:
            axis_colors = dict(x_color=(1.0, 0.65, 0.65), y_color=(0.65, 1.0, 0.7), z_color=(0.65, 0.75, 1.0))
        else:
            # Pastel, but a touch more saturated than the dark-theme
            # trio (same convention as ACCENT's light/dark pair) so they
            # still read clearly against a white/cream background.
            axis_colors = dict(x_color=(0.85, 0.35, 0.35), y_color=(0.25, 0.65, 0.35), z_color=(0.30, 0.45, 0.80))

        xlabel, ylabel, zlabel = self.axis_labels
        self.axes_actor = self.plotter.add_axes(
            line_width=2,
            labels_off=False,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            color=label_rgb,
            **axis_colors,
        )
        for caption in (
            self.axes_actor.GetXAxisCaptionActor2D(),
            self.axes_actor.GetYAxisCaptionActor2D(),
            self.axes_actor.GetZAxisCaptionActor2D(),
        ):
            text_prop = caption.GetCaptionTextProperty()
            text_prop.SetFontFamilyToTimes()
            text_prop.SetColor(*label_rgb)
            text_prop.SetShadow(False)

        # add_axes() always (re-)enables the orientation widget — reapply
        # the user's own Mini Axes toggle state.
        if not self.mini_axes_visible:
            self.plotter.hide_axes()

    def set_mini_axes_visible(self, visible: bool):
        """Show/hide the small bottom-left orientation-widget triad."""
        self.mini_axes_visible = bool(visible)
        if self.mini_axes_visible:
            self.plotter.show_axes()
        else:
            self.plotter.hide_axes()
        self.plotter.render()

    def _build_cube_axes(self):
        """Build the cube-face axes actor backing the Main Axes Labels /
        Ticks and Labels / Grid Lines toggles. Camera-linked fly mode
        ("closest triad") is what gives the classic dynamic behaviour —
        VTK always draws just the 3 nearest edges (usually the bottom
        ones) and re-picks them as the view rotates, rather than
        cluttering all 12 edges."""
        from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor

        self.cube_axes_actor = vtkCubeAxesActor()
        self.cube_axes_actor.SetBounds(*self.grid.bounds)
        self.cube_axes_actor.SetCamera(self.plotter.renderer.GetActiveCamera())
        self.cube_axes_actor.SetFlyModeToClosestTriad()
        self.cube_axes_actor.SetVisibility(False)
        # VTK silently rescales large-magnitude labels (e.g. "3600" becomes
        # "3.6" with a "(x10^3)" suffix on the title) regardless of the
        # LabelFormat string — disable that so the axis_label_formats we
        # set below are actually what gets drawn, full-magnitude.
        self.cube_axes_actor.SetLabelScaling(False, 0, 0, 0)
        # Extra breathing room between the tick labels and the axis title
        # text (default is (20, 20)).
        self.cube_axes_actor.SetTitleOffset((36.0, 36.0))
        # The geometric bounds above are in voxel space (and the Z axis is
        # additionally stretched by vel_scale purely for a cube-ish aspect
        # ratio) — SetAxisRange remaps what VALUES are printed at those
        # same geometric positions, so ticks/labels read in real RA/Dec/
        # velocity (or kpc) rather than raw voxel indices, independent of
        # that stretching.
        if all(r is not None for r in self.axis_ranges):
            (x0, x1), (y0, y1), (z0, z1) = self.axis_ranges
            self.cube_axes_actor.SetXAxisRange(x0, x1)
            self.cube_axes_actor.SetYAxisRange(y0, y1)
            self.cube_axes_actor.SetZAxisRange(z0, z1)
        self.plotter.renderer.AddActor(self.cube_axes_actor)

    def set_main_axes_labels_visible(self, visible: bool):
        """Toggle the axis *name* labels (e.g. "RA"/"Dec"/"km/s") on the
        main cube's own 3 dynamically-chosen edges — independent of the
        numeric tick marks/values (see set_axis_ticks_visible)."""
        self.show_main_axes_labels = bool(visible)
        self._style_cube_axes()
        self.plotter.render()

    def set_axis_ticks_visible(self, visible: bool):
        """Toggle tick marks + numeric tick-value labels on the main
        cube's 3 dynamically-chosen edges."""
        self.show_axis_ticks = bool(visible)
        self._style_cube_axes()
        self.plotter.render()

    def set_grid_lines_visible(self, visible: bool):
        """Toggle faint (low-opacity) grid lines spanning the volume."""
        self.show_grid_lines = bool(visible)
        self._style_cube_axes()
        self.plotter.render()

    def set_colorbar_visible(self, visible: bool):
        """Toggle the top-right colorbar (swatch + border/ticks + title)."""
        self.colorbar_visible = bool(visible)
        for actor in (
            self._colorbar_actor,
            self._colorbar_frame_actor,
            self._colorbar_tick_actor,
            self._colorbar_title_actor,
            *self._colorbar_value_label_actors,
        ):
            if actor is not None:
                actor.SetVisibility(self.colorbar_visible)
        self.plotter.render()

    def set_scalebar_visible(self, visible: bool):
        """Toggle the top-left scale bar (bracket + length label)."""
        self.scalebar_visible = bool(visible)
        for actor in (self._scale_bar_actor2d, self._scale_bar_text_actor):
            if actor is not None:
                actor.SetVisibility(self.scalebar_visible)
        self.plotter.render()

    def set_spatial_scale(self, value_per_voxel: float, unit: str):
        """Supply (or replace) the physical length a voxel spans, e.g.
        after the user fills in "Field of view" for a numpy cube that
        was loaded with no spatial scale at all — the scale bar actors
        are only ever built once a scale exists in the first place."""
        self.spatial_scale = (float(value_per_voxel), unit)
        if self._scale_bar_actor2d is None:
            self._build_scale_bar()
            self._style_scale_bar()
        self._update_scale_bar()
        self.plotter.render()

    def set_manual_axis_scale(self, axis_idx: int, value_per_voxel: float, unit_text: str, centered: bool = False):
        """Give a numpy-array cube's otherwise-unitless axis (raw voxel-
        index ticks) real physical units, once the user has supplied
        Field of view (for the spatial axes) or Spectral Resolution (for
        the velocity axis) plus a unit — the same custom tick/title
        machinery already used for FITS/HDF5 cubes' RA/Dec/velocity
        axes, just populated interactively rather than at load time.
        ``unit_text`` is shown, LaTeX-rendered (matplotlib mathtext), in
        parentheses after that axis's title, e.g. "X ($M_\\odot/h$)".
        ``centered``: a spatial (Field of view) axis's ticks run 0 -> N
        from one vertex of the cube; the velocity (Spectral Resolution)
        axis instead centres on systemic velocity, -N/2 -> N/2 — pass
        True for that one."""
        n_voxels = self.grid.dimensions[axis_idx]
        full = value_per_voxel * n_voxels
        if centered:
            half = full / 2.0
            self.axis_ranges[axis_idx] = (-half, half)
        else:
            self.axis_ranges[axis_idx] = (0.0, full)
        self.axis_tick_formatters[axis_idx] = lambda v: f"{v:.2f}"
        self.axis_tick_units[axis_idx] = None
        self._manual_axis_units[axis_idx] = unit_text
        self._style_cube_axes()
        self._rebuild_custom_tick_labels(force=True)
        self._rebuild_custom_titles(force=True)
        self.plotter.render()

    def clear_manual_axis_scale(self, axis_idx: int):
        """Revert one axis back to raw voxel-index ticks — used when the
        user clears its Field of view/unit or Spectral Resolution/unit
        fields, or switches Type of cube to PPP (no velocity axis)."""
        self.axis_ranges[axis_idx] = None
        self.axis_tick_formatters[axis_idx] = None
        self._manual_axis_units[axis_idx] = None
        self._style_cube_axes()
        self._rebuild_custom_tick_labels(force=True)
        self._rebuild_custom_titles(force=True)
        self.plotter.render()

    def set_colorbar_title(self, title: str):
        """Change the colorbar's title text, e.g. once the user fills in
        "Quantity Units" for a numpy cube that started with a blank
        title."""
        self.colorbar_title = title
        self._rebuild_colorbar()
        self.plotter.render()

    def _cube_outline_mesh(self, style: str):
        """Build the wireframe box mesh for the given line style. "solid"
        just reuses the grid's own continuous-edge outline; "dashed" and
        "dotted" are built as many short line segments instead, since
        vtkProperty's line-stipple pattern has no effect under VTK's
        modern OpenGL2 rendering backend."""
        if style == "solid":
            return self.grid.outline()

        x0, x1, y0, y1, z0, z1 = self.grid.bounds
        corners = {
            0: (x0, y0, z0), 1: (x1, y0, z0), 2: (x1, y1, z0), 3: (x0, y1, z0),
            4: (x0, y0, z1), 5: (x1, y0, z1), 6: (x1, y1, z1), 7: (x0, y1, z1),
        }
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        n_per_edge, dash_frac = (18, 0.45) if style == "dashed" else (40, 0.12)

        segments = []
        for a, b in edges:
            pa, pb = np.array(corners[a]), np.array(corners[b])
            step = (pb - pa) / n_per_edge
            for i in range(n_per_edge):
                start = pa + step * i
                end = start + step * dash_frac
                segments.append(pv.Line(start, end))
        mesh = segments[0]
        for seg in segments[1:]:
            mesh = mesh.merge(seg)
        return mesh

    def _rebuild_cube_outline(self):
        mesh = self._cube_outline_mesh(self.cube_outline_style)
        if self.bbox_actor is not None:
            self.plotter.remove_actor(self.bbox_actor)
        fg_color = "white" if self.is_dark_theme else "black"
        self.bbox_actor = self.plotter.add_mesh(mesh, line_width=self.cube_outline_thickness, color=fg_color)
        self.bbox_actor.SetVisibility(self.cube_outline_visible)

    def set_cube_outline_visible(self, visible: bool):
        """Toggle the wireframe box outlining the volume's bounds."""
        self.cube_outline_visible = bool(visible)
        if self.bbox_actor is not None:
            self.bbox_actor.SetVisibility(self.cube_outline_visible)
        self.plotter.render()

    def set_cube_outline_thickness(self, width: float):
        self.cube_outline_thickness = width
        if self.bbox_actor is not None:
            self.bbox_actor.prop.line_width = width
        self.plotter.render()

    def set_cube_outline_style(self, style: str):
        self.cube_outline_style = style
        self._rebuild_cube_outline()
        self.plotter.render()

    def _style_cube_axes(self):
        actor = self.cube_axes_actor
        if actor is None:
            return

        any_on = self.show_main_axes_labels or self.show_axis_ticks or self.show_grid_lines
        actor.SetVisibility(any_on)

        # An axis with its own custom title overlay (see
        # _rebuild_custom_titles) always keeps vtkCubeAxesActor's native
        # title blank — our own text replaces it entirely, for the same
        # font/rotation consistency as the custom tick labels.
        titles = tuple(
            "" if self.axis_tick_formatters[i] else (self.axis_labels[i] if self.show_main_axes_labels else "")
            for i in range(3)
        )
        actor.SetXTitle(titles[0])
        actor.SetYTitle(titles[1])
        actor.SetZTitle(titles[2])

        # VTK ties the axis *title*'s rendering to AxisLabelVisibility
        # internally, so that flag has to stay on whenever a title should
        # show — even if the numeric tick values themselves (Ticks and
        # Labels) are off. There's no separate "title only" flag, and an
        # empty/blank SetXLabelFormat is silently ignored (VTK falls back
        # to its default numeric format regardless) — so the actual fix is
        # to keep LabelVisibility on but make the tick-*value* text itself
        # invisible via its own text property's opacity, leaving the
        # title's separate text property untouched. Tick *marks* (short
        # perpendicular lines) are unaffected by any of this — purely
        # show_axis_ticks.
        label_visibility_needed = self.show_main_axes_labels or self.show_axis_ticks
        for i, axis in enumerate("XYZ"):
            getattr(actor, f"Set{axis}LabelFormat")(self.axis_label_formats[i])
        for i, axis in enumerate("XYZ"):
            has_custom = bool(self.axis_tick_formatters[i])
            getattr(actor, f"Set{axis}AxisVisibility")(any_on)
            # An axis with its own custom tick-mark/gridline overlay (see
            # _rebuild_custom_tick_labels) always keeps vtkCubeAxesActor's
            # native tick marks and gridlines off — they're computed from
            # VTK's own internal (and unreadable) tick count, which almost
            # never matches our custom tick positions, producing visibly
            # misaligned marks/gridlines if both were drawn.
            getattr(actor, f"Set{axis}AxisTickVisibility")(False if has_custom else self.show_axis_ticks)
            getattr(actor, f"Set{axis}AxisMinorTickVisibility")(False)  # major ticks only
            getattr(actor, f"Set{axis}AxisLabelVisibility")(label_visibility_needed)
            draw_native_grid = self.show_grid_lines and not has_custom
            getattr(actor, f"Draw{axis}Gridlines{'On' if draw_native_grid else 'Off'}")()

        label_rgb = self._theme_rgb()
        label_getters = (actor.GetXAxesLabelProperty, actor.GetYAxesLabelProperty, actor.GetZAxesLabelProperty)
        for i, getter in enumerate(label_getters):
            prop = getter()
            prop.SetColor(*label_rgb)
            prop.SetFontFamilyToTimes()
            # An axis with its own custom tick-label overlay (see
            # _rebuild_custom_tick_labels) always keeps VTK's native
            # numeric text hidden — our own text replaces it entirely
            # rather than being combined with it.
            if self.axis_tick_formatters[i]:
                prop.SetOpacity(0.0)
            else:
                prop.SetOpacity(1.0 if self.show_axis_ticks else 0.0)

        for getter in (actor.GetXAxesTitleProperty, actor.GetYAxesTitleProperty, actor.GetZAxesTitleProperty):
            prop = getter()
            prop.SetColor(*label_rgb)
            prop.SetFontFamilyToTimes()

        # Faint grid lines: low opacity so they read as a subtle
        # reference frame rather than clutter.
        for getter in (actor.GetXAxesGridlinesProperty, actor.GetYAxesGridlinesProperty, actor.GetZAxesGridlinesProperty):
            prop = getter()
            prop.SetColor(*label_rgb)
            prop.SetOpacity(0.15)

        if any(self.axis_tick_formatters):
            self._rebuild_custom_tick_labels(force=True)
            self._rebuild_custom_titles(force=True)

    def _closest_triad_edge(self, axis_idx: int):
        """World-space (p0, p1) endpoints of whichever of the 4 parallel
        edges vtkCubeAxesActor's own "closest triad" fly mode would pick
        for this axis direction — i.e. the edge on the side of the box
        actually facing the camera, so our custom labels track the same
        edge its native tick marks/gridlines are drawn on. There's no
        public API to read VTK's own choice back, so this replicates it
        with the standard heuristic: for each of the *other* two axes,
        pick whichever bound is closer to the camera."""
        b = self.grid.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        cam_pos = np.array(self.plotter.renderer.GetActiveCamera().GetPosition())
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])

        def pick(lo, hi, cam_c, center_c):
            return hi if cam_c > center_c else lo

        if axis_idx == 0:  # X varies; fix Y, Z
            y = pick(b[2], b[3], cam_pos[1], center[1])
            z = pick(b[4], b[5], cam_pos[2], center[2])
            return np.array([b[0], y, z]), np.array([b[1], y, z])
        elif axis_idx == 1:  # Y varies; fix X, Z
            x = pick(b[0], b[1], cam_pos[0], center[0])
            z = pick(b[4], b[5], cam_pos[2], center[2])
            return np.array([x, b[2], z]), np.array([x, b[3], z])
        else:  # Z varies; fix X, Y
            x = pick(b[0], b[1], cam_pos[0], center[0])
            y = pick(b[2], b[3], cam_pos[1], center[1])
            return np.array([x, y, b[4]]), np.array([x, y, b[5]])

    def _edge_outward_direction(self, axis_idx: int, p0, center) -> np.ndarray:
        """Unit vector pointing from the box centre out through edge
        endpoint `p0`, in the two dimensions *other* than `axis_idx` (its
        own varying dimension is zeroed) — i.e. straight out through the
        cube's corner, so a label offset along it sits outside the
        surface rather than sliding along the edge."""
        d = np.array(p0, dtype=float) - center
        d[axis_idx] = 0.0
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(3)

    def _target_tick_count(self, axis_idx: int) -> int:
        """Fewer ticks when zoomed out, more when zoomed in — a simple
        zoom-adaptive density VTK's own fixed-count label system can't do
        (see axis_tick_formatters docstring). The velocity axis holds a
        fixed density instead of zoom-adapting."""
        if axis_idx == 2:
            return 5
        camera = self.plotter.renderer.GetActiveCamera()
        distance = camera.GetDistance()
        if self._camera_ref_distance is None:
            self._camera_ref_distance = distance
        ratio = self._camera_ref_distance / max(distance, 1e-6)
        if ratio > 2.5:
            return 7
        if ratio > 1.5:
            return 5
        return 3

    def _make_line_actor(self, p_a, p_b, color, opacity=1.0, line_width=1.0):
        from vtkmodules.vtkFiltersSources import vtkLineSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper

        src = vtkLineSource()
        src.SetPoint1(*p_a)
        src.SetPoint2(*p_b)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(src.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        actor.GetProperty().SetLineWidth(line_width)
        return actor

    def _face_gridlines_for_tick(self, axis_idx: int, edge_point, p0, b):
        """The two line segments (on the pair of cube faces meeting at
        this axis's picked edge) that pass through `edge_point` — replaces
        vtkCubeAxesActor's native gridlines, which are drawn from its own
        internal tick positions and so don't line up with our custom
        ticks (see _rebuild_custom_tick_labels)."""
        other_dims = [d for d in (0, 1, 2) if d != axis_idx]
        segments = []
        for varying_dim in other_dims:
            p_a = list(edge_point)
            p_b = list(edge_point)
            p_a[varying_dim] = b[2 * varying_dim]
            p_b[varying_dim] = b[2 * varying_dim + 1]
            segments.append((p_a, p_b))
        return segments

    def _rebuild_custom_tick_labels(self, force: bool = False):
        """(Re)draw the custom RA/Dec/velocity tick-label overlay, plus
        its own tick marks and (when Grid Lines is on) gridlines at the
        exact same positions (see axis_tick_formatters and
        _face_gridlines_for_tick) — vtkCubeAxesActor's native marks/
        gridlines are always suppressed for these axes (see
        _style_cube_axes) since they come from its own internal tick
        count, which almost never matches ours. Cheap to call often: the
        (edge, tick count) state key is recomputed first, and the actual
        actors are only torn down and rebuilt when that key actually
        changes — so camera-drag callbacks stay lightweight. Orientation
        /scale of the *text* (see _update_custom_tick_transforms) is
        refreshed separately, on every camera move, since that must track
        continuously; the tick marks/gridlines are plain 3D geometry and
        don't need it."""
        show = (
            (self.show_axis_ticks or self.show_grid_lines)
            and self.cube_axes_actor is not None
            and self.cube_axes_actor.GetVisibility()
        )
        if not show:
            if self._custom_tick_actors or self._custom_tickmark_actors or self._custom_gridline_actors:
                for a in self._custom_tick_actors + self._custom_tickmark_actors + self._custom_gridline_actors:
                    self.plotter.renderer.RemoveActor(a)
                self._custom_tick_actors = []
                self._custom_tick_edge_dirs = []
                self._custom_tickmark_actors = []
                self._custom_gridline_actors = []
                self._custom_tick_state_key = None
                self.plotter.render()
            return

        edges = {i: self._closest_triad_edge(i) for i in (0, 1, 2) if self.axis_tick_formatters[i]}
        targets = {i: self._target_tick_count(i) for i in edges}
        key = (
            self.show_grid_lines,
            tuple(targets.values()),
            tuple(tuple(np.round(p, 4)) for pair in edges.values() for p in pair),
        )
        if not force and key == self._custom_tick_state_key:
            return
        self._custom_tick_state_key = key

        from vtkmodules.vtkRenderingCore import vtkTextActor3D

        for a in self._custom_tick_actors + self._custom_tickmark_actors + self._custom_gridline_actors:
            self.plotter.renderer.RemoveActor(a)
        self._custom_tick_actors = []
        self._custom_tick_edge_dirs = []
        self._custom_tickmark_actors = []
        self._custom_gridline_actors = []

        b = self.grid.bounds
        diag = math.dist((b[0], b[2], b[4]), (b[1], b[3], b[5]))
        # Pushed out just enough to clear the tick marks, with the axis
        # *title* text sitting further out again (see
        # _rebuild_custom_titles) for visible separation between the two.
        offset_mag = diag * 0.025
        tickmark_len = diag * 0.012
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        label_rgb = self._theme_rgb()

        for axis_idx, (p0, p1) in edges.items():
            formatter = self.axis_tick_formatters[axis_idx]
            units = self.axis_tick_units[axis_idx]
            vmin, vmax = self.axis_ranges[axis_idx]
            edge_vec = np.array(p1) - np.array(p0)
            edge_dir = edge_vec / np.linalg.norm(edge_vec)
            outward_dir = self._edge_outward_direction(axis_idx, p0, center)
            outward = outward_dir * offset_mag

            # _nice_ticks needs an ascending span — vmin/vmax themselves
            # may be given in either order (e.g. velocity decreasing with
            # channel index), which the frac interpolation below already
            # handles correctly regardless of direction.
            ticks = _nice_ticks(min(vmin, vmax), max(vmin, vmax), target=targets[axis_idx])
            if units is not None:
                components = [formatter(v) for v in ticks]
                strings = _format_ticks_with_abbreviation(components, units)
            else:
                # Plain numeric axis (e.g. velocity) — no sexagesimal
                # component abbreviation, just the formatter's own text.
                strings = [formatter(v) for v in ticks]

            for v, text in zip(ticks, strings):
                frac = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
                edge_point = np.array(p0) + frac * edge_vec

                if self.show_axis_ticks:
                    point = edge_point + outward
                    actor = vtkTextActor3D()
                    actor.SetInput(text)
                    prop = actor.GetTextProperty()
                    prop.SetColor(*label_rgb)
                    prop.SetFontFamilyToTimes()
                    prop.SetFontSize(self._label_font_size + 2)
                    prop.SetJustificationToCentered()
                    prop.SetVerticalJustificationToCentered()
                    self.plotter.renderer.AddActor(actor)
                    self._custom_tick_actors.append(actor)
                    self._custom_tick_edge_dirs.append((actor, tuple(point), edge_dir))

                    mark_actor = self._make_line_actor(
                        edge_point, edge_point - outward_dir * tickmark_len, label_rgb
                    )
                    self.plotter.renderer.AddActor(mark_actor)
                    self._custom_tickmark_actors.append(mark_actor)

                if self.show_grid_lines:
                    for p_a, p_b in self._face_gridlines_for_tick(axis_idx, edge_point, p0, b):
                        grid_actor = self._make_line_actor(p_a, p_b, label_rgb, opacity=0.15)
                        self.plotter.renderer.AddActor(grid_actor)
                        self._custom_gridline_actors.append(grid_actor)
        self._update_custom_tick_transforms()
        self.plotter.render()

    def _build_mathtext_actor(self, text: str, color_rgb, font_size: int):
        """A textured billboard quad showing LaTeX-ish mathtext (see
        _render_mathtext_rgba) — used in place of vtkTextActor3D (which
        can only draw plain text) for axis titles that include a
        physical unit. Sized in the same "1 texture pixel = 1 world
        unit at scale=1" convention vtkTextActor3D uses internally, so
        it billboards/rescales correctly through the exact same
        _update_custom_tick_transforms() matrix math and shared
        (actor, point, edge_dir) list as the plain-text title actors."""
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkFiltersSources import vtkPlaneSource
        from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkTexture
        from vtkmodules.util import numpy_support

        rgba = _render_mathtext_rgba(text, color=color_rgb, fontsize=font_size)
        h, w = rgba.shape[:2]

        image = vtkImageData()
        image.SetDimensions(w, h, 1)
        flat = np.flipud(rgba).reshape(-1, 4)  # VTK images are bottom-up
        image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat, deep=True))

        texture = vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOn()

        # _render_mathtext_rgba rasterizes at _MATHTEXT_DPI (200) for
        # crisp glyph edges, so its bitmap is (200/72) taller/wider in
        # pixels than vtkTextActor3D's own "1 texture pixel = 1 world
        # unit" bitmap would be for the same nominal font_size (VTK's
        # convention implicitly assumes ~72 dpi) — without correcting
        # for that, mathtext titles render ~2.8x too large next to
        # plain-text ones at the same font_size. Scale the plane (not
        # the texture) down to compensate, keeping the supersampled
        # texture for quality while matching the intended world size.
        plane_w = w * 72.0 / _MATHTEXT_DPI
        plane_h = h * 72.0 / _MATHTEXT_DPI

        plane = vtkPlaneSource()
        plane.SetOrigin(-plane_w / 2.0, -plane_h / 2.0, 0.0)
        plane.SetPoint1(plane_w / 2.0, -plane_h / 2.0, 0.0)
        plane.SetPoint2(-plane_w / 2.0, plane_h / 2.0, 0.0)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.SetTexture(texture)
        # Flat/unlit, like text — scene lighting would otherwise shade
        # the label darker depending on viewing angle.
        actor.GetProperty().LightingOff()
        return actor

    def _build_mathtext_actor2d(self, text: str, color_rgb, font_size: int, anchor_x: float, anchor_y: float):
        """A 2D (fixed screen-space) mathtext image, used for the
        colorbar title — unlike _build_mathtext_actor's 3D billboard for
        axis titles, this stays put in normalized viewport coordinates.
        ``(anchor_x, anchor_y)`` is where the image's horizontal centre/
        vertical bottom should land, matching the plain vtkTextActor
        this replaces (SetJustificationToCentered + default bottom
        vertical justification)."""
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkImageMapper
        from vtkmodules.util import numpy_support

        rgba = _render_mathtext_rgba(text, color=color_rgb, fontsize=font_size)
        h, w = rgba.shape[:2]

        image = vtkImageData()
        image.SetDimensions(w, h, 1)
        flat = np.flipud(rgba).reshape(-1, 4)
        image.GetPointData().SetScalars(numpy_support.numpy_to_vtk(flat, deep=True))

        mapper = vtkImageMapper()
        mapper.SetInputData(image)
        mapper.SetColorWindow(255)
        mapper.SetColorLevel(127.5)

        actor = vtkActor2D()
        actor.SetMapper(mapper)
        # vtkImageMapper anchors an actor's *bottom-left* corner at its
        # position — shift left by half the image's width (converted to
        # normalized-viewport units) so anchor_x lands at the horizontal
        # centre instead.
        window_size = self.plotter.renderer.GetRenderWindow().GetSize()
        win_w = max(window_size[0], 1)
        actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        actor.SetPosition(anchor_x - (w / 2.0) / win_w, anchor_y)
        return actor

    def _rebuild_custom_titles(self, force: bool = False):
        """(Re)draw the custom axis-title overlay ("RA"/"Dec"/"km/s") —
        the exact same vtkTextActor3D + billboard-that-follows-the-edge
        mechanism as the tick labels (see _rebuild_custom_tick_labels),
        just one label per axis, larger, and sitting further out so
        there's clear separation from the tick-value text inside it."""
        show = self.show_main_axes_labels and self.cube_axes_actor is not None
        if not show:
            if self._custom_title_actors:
                for a in self._custom_title_actors:
                    self.plotter.renderer.RemoveActor(a)
                self._custom_title_actors = []
                self._custom_title_edge_dirs = []
                self._custom_title_state_key = None
                self.plotter.render()
            return

        edges = {i: self._closest_triad_edge(i) for i in (0, 1, 2) if self.axis_tick_formatters[i]}
        key = tuple(tuple(np.round(p, 4)) for pair in edges.values() for p in pair)
        if not force and key == self._custom_title_state_key:
            return
        self._custom_title_state_key = key

        from vtkmodules.vtkRenderingCore import vtkTextActor3D

        for a in self._custom_title_actors:
            self.plotter.renderer.RemoveActor(a)
        self._custom_title_actors = []
        self._custom_title_edge_dirs = []

        b = self.grid.bounds
        diag = math.dist((b[0], b[2], b[4]), (b[1], b[3], b[5]))
        offset_mag = diag * 0.055  # further out than the tick labels' 0.025, but not by much
        center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        label_rgb = self._theme_rgb()

        for axis_idx, (p0, p1) in edges.items():
            text = self.axis_labels[axis_idx]
            if not text:
                continue
            edge_vec = np.array(p1) - np.array(p0)
            edge_dir = edge_vec / np.linalg.norm(edge_vec)
            outward = self._edge_outward_direction(axis_idx, p0, center) * offset_mag
            point = np.array(p0) + 0.5 * edge_vec + outward

            unit_text = self._manual_axis_units[axis_idx]
            if unit_text:
                actor = self._build_mathtext_actor(f"{text} ({unit_text})", label_rgb, self._label_font_size + 4)
            else:
                actor = vtkTextActor3D()
                actor.SetInput(text)
                prop = actor.GetTextProperty()
                prop.SetColor(*label_rgb)
                prop.SetFontFamilyToTimes()
                prop.SetFontSize(self._label_font_size + 4)
                prop.BoldOn()
                prop.SetJustificationToCentered()
                prop.SetVerticalJustificationToCentered()
            self.plotter.renderer.AddActor(actor)
            self._custom_title_actors.append(actor)
            self._custom_title_edge_dirs.append((actor, tuple(point), edge_dir))

        self._update_custom_tick_transforms()
        self.plotter.render()

    def _update_custom_tick_transforms(self):
        """Re-orient every custom tick-label and title actor to face the
        camera while its text baseline still follows its cube edge (the
        same "billboard that also rotates with the axis" behaviour
        vtkAxisFollower gives VTK's own native labels — vtkTextActor3D has
        no built-in equivalent, since unlike vtkVectorText it supports a
        real font), and rescale it to hold a constant on-screen size as
        the camera moves. Cheap (no actor creation), so this runs on every
        camera-modified event rather than being throttled like the
        rebuilds above."""
        entries = self._custom_tick_edge_dirs + self._custom_title_edge_dirs
        if not entries:
            return
        from vtkmodules.vtkCommonMath import vtkMatrix4x4

        camera = self.plotter.renderer.GetActiveCamera()
        vpn = np.array(camera.GetViewPlaneNormal())
        vup = np.array(camera.GetViewUp())
        right = np.cross(vup, vpn)
        rnorm = np.linalg.norm(right)
        if rnorm < 1e-9:
            return
        right = right / rnorm
        true_up = np.cross(vpn, right)

        window_size = self.plotter.renderer.GetRenderWindow().GetSize()
        height_px = max(window_size[1], 1)
        if camera.GetParallelProjection():
            world_per_px = (2.0 * camera.GetParallelScale()) / height_px
        else:
            view_angle_rad = math.radians(camera.GetViewAngle())
            distance = camera.GetDistance()
            world_per_px = (2.0 * distance * math.tan(view_angle_rad / 2.0)) / height_px
        # vtkTextActor3D's un-scaled quad is already sized to match its
        # rendered bitmap 1:1 (one texture pixel = one world unit) — since
        # that bitmap is rendered at FontSize, a plain `world_per_px`
        # scale factor (not multiplied by FontSize again) is what keeps
        # the on-screen size equal to FontSize screen pixels, constant
        # regardless of zoom.
        scale = world_per_px

        for actor, point, edge_dir in entries:
            proj = edge_dir - np.dot(edge_dir, vpn) * vpn
            pnorm = np.linalg.norm(proj)
            x_axis = (proj / pnorm) if pnorm > 1e-6 else right
            if np.dot(x_axis, right) < 0:
                x_axis = -x_axis
            y_axis = np.cross(vpn, x_axis)
            if np.dot(y_axis, true_up) < 0:
                x_axis, y_axis = -x_axis, -y_axis

            matrix = vtkMatrix4x4()
            for row in range(3):
                matrix.SetElement(row, 0, x_axis[row] * scale)
                matrix.SetElement(row, 1, y_axis[row] * scale)
                matrix.SetElement(row, 2, vpn[row] * scale)
                matrix.SetElement(row, 3, point[row])
            actor.SetUserMatrix(matrix)

    def _maybe_rebuild_custom_tick_labels(self):
        try:
            self._rebuild_custom_tick_labels(force=False)
            self._rebuild_custom_titles(force=False)
            self._update_custom_tick_transforms()
        except Exception:
            pass

    def _build_scale_bar(self):
        """Create the fixed-on-screen-length scale bar (top-left): a
        slightly thicker line with small end-ticks (like a "|——|"
        bracket) plus a text label, centred over the bar, reporting the
        physical length it currently represents. The bar's on-screen
        size never changes with zoom — only the label text does (see
        _update_scale_bar)."""
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkCoordinate, vtkPolyDataMapper2D, vtkTextActor

        y = 0.90
        x0 = 0.03
        x1 = x0 + self._scale_bar_frac
        tick_half = 0.006  # end-tick half-height, in normalized viewport units

        points = vtkPoints()
        # 0,1: main horizontal bar; 2,3: left end-tick; 4,5: right end-tick
        points.InsertNextPoint(x0, y, 0.0)
        points.InsertNextPoint(x1, y, 0.0)
        points.InsertNextPoint(x0, y - tick_half, 0.0)
        points.InsertNextPoint(x0, y + tick_half, 0.0)
        points.InsertNextPoint(x1, y - tick_half, 0.0)
        points.InsertNextPoint(x1, y + tick_half, 0.0)

        cells = vtkCellArray()
        for a, b in ((0, 1), (2, 3), (4, 5)):
            line = vtkLine()
            line.GetPointIds().SetId(0, a)
            line.GetPointIds().SetId(1, b)
            cells.InsertNextCell(line)
        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(cells)

        coord = vtkCoordinate()
        coord.SetCoordinateSystemToNormalizedViewport()
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        mapper.SetTransformCoordinate(coord)

        self._scale_bar_actor2d = vtkActor2D()
        self._scale_bar_actor2d.SetMapper(mapper)
        self._scale_bar_actor2d.GetProperty().SetLineWidth(3)
        self.plotter.renderer.AddActor2D(self._scale_bar_actor2d)

        self._scale_bar_text_actor = vtkTextActor()
        self._scale_bar_text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self._scale_bar_text_actor.SetPosition((x0 + x1) / 2, y + 0.015)
        text_prop = self._scale_bar_text_actor.GetTextProperty()
        text_prop.SetFontFamilyToTimes()
        text_prop.SetFontSize(self._label_font_size)
        text_prop.SetJustificationToCentered()
        text_prop.SetBold(True)
        text_prop.SetShadow(False)
        self.plotter.renderer.AddActor2D(self._scale_bar_text_actor)

        if self.spatial_scale is None:
            self._scale_bar_text_actor.SetInput("px")

    def _style_scale_bar(self):
        if self._scale_bar_text_actor is None:
            return
        rgb = (1.0, 1.0, 1.0) if self.is_dark_theme else (0.0, 0.0, 0.0)
        self._scale_bar_text_actor.GetTextProperty().SetColor(*rgb)
        self._scale_bar_actor2d.GetProperty().SetColor(*rgb)

    # Unit -> display symbol/word. Arcseconds use the conventional double-
    # quote symbol; physical units are spelled out.
    _UNIT_SYMBOLS = {"arcsec": '"'}

    def _update_scale_bar(self):
        """Recompute the physical length the (fixed on-screen) scale bar
        currently represents — called on every camera change (pan, zoom,
        rotate), from any input source (trackpad, mouse, or programmatic).
        When the spatial scale is unknown, the bar just reads "px" and
        never changes (there's no physical quantity to update)."""
        if self._scale_bar_text_actor is None:
            return
        if self.spatial_scale is None:
            return
        scale_per_voxel, unit = self.spatial_scale
        renderer = self.plotter.renderer
        camera = renderer.GetActiveCamera()
        render_window = renderer.GetRenderWindow()
        # During teardown (e.g. the render window is mid-close while a
        # queued camera ModifiedEvent still fires) this can legitimately
        # be None — nothing to update in that case.
        if render_window is None:
            return
        size = render_window.GetSize()
        if not size or size[0] == 0 or size[1] == 0:
            return

        pos = np.array(camera.GetPosition())
        fp = np.array(camera.GetFocalPoint())
        distance = np.linalg.norm(fp - pos)
        if camera.GetParallelProjection():
            world_height = camera.GetParallelScale() * 2
        else:
            angle = np.radians(camera.GetViewAngle())
            world_height = 2 * distance * np.tan(angle / 2)
        world_units_per_px = world_height / size[1]

        bar_px = self._scale_bar_frac * size[0]
        physical_length = world_units_per_px * bar_px * scale_per_voxel
        symbol = self._UNIT_SYMBOLS.get(unit)
        text = f"{physical_length:.2g}{symbol}" if symbol else f"{physical_length:.2g} {unit}"
        self._scale_bar_text_actor.SetInput(text)

    def redraw_volume(self, render: bool = True):
        # PyVista's remove_actor()/add_volume(name=...) both go through its
        # own actor-tracking dict (Renderer._actors), which under some
        # versions/call sequences ends up deleted (see Renderer.close())
        # while an actor removal is still in flight, raising
        # AttributeError. Removing the previous volume via the raw VTK
        # renderer call instead sidesteps that bookkeeping entirely.
        if getattr(self, "_volume_actor", None) is not None:
            try:
                self.plotter.renderer.RemoveVolume(self._volume_actor)
            except Exception:
                pass

        if self.value_scale == "log":
            scalars_name = "intensity_log"
            lo = max(self.current_clim[0], self._log_floor)
            hi = max(self.current_clim[1], self._log_floor)
            if hi <= lo:
                hi = lo * 10.0
            clim = [math.log10(lo), math.log10(hi)]
        else:
            scalars_name = "intensity"
            clim = self.current_clim

        self._volume_actor = self.plotter.add_volume(
            self.grid,
            scalars=scalars_name,
            cmap=self.cmap,
            opacity=self.opacity,
            clim=clim,
            show_scalar_bar=False,
            # Removing the old volume via raw VTK above (rather than
            # pyvista's own tracked remove_actor) makes pyvista think this
            # is the first actor ever added, so its "reset_camera=None"
            # auto-detection kicks in and silently resets the camera/pivot
            # on every redraw (e.g. every theme toggle, slider drag, or
            # colormap change) — pin it off explicitly.
            reset_camera=False,
            render=False,
        )
        self._volume_actor.prop.interpolation_type = "linear"

        if not self.show_moment0:
            self._rebuild_colorbar()

        if render:
            self.plotter.render()

    _COLORBAR_N_LABELS = 4

    def _rebuild_colorbar(self):
        """Top-right colorbar: a border around just the gradient swatch
        (not the title/labels), small ticks in the same colour as that
        border, tick labels sized to match the scale bar text, extra
        padding between the ticks and the swatch, and a title taken from
        the cube's own intensity unit (e.g. FITS BUNIT) rather than a
        generic "Intensity"."""
        try:
            if self._colorbar_actor is not None:
                self.plotter.remove_scalar_bar()
        except Exception:
            pass

        text_color = "white" if self.is_dark_theme else "black"
        cb_x, cb_y, cb_w, cb_h = 0.66, 0.86, 0.26, 0.07

        # add_scalar_bar() unconditionally reads mapper.lookup_table
        # (== mapper._lut) — pyvista sets that as a side effect of
        # add_volume(), but it isn't reliably still present on every
        # later call here (observed going missing between an earlier
        # successful rebuild and a subsequent one, cause not fully
        # pinned down further than "a pyvista/VTK volume-mapper
        # bookkeeping quirk"); rebuilding it explicitly if absent avoids
        # an AttributeError that otherwise silently drops the colorbar.
        mapper = self._volume_actor.mapper
        if getattr(mapper, "_lut", None) is None:
            # Built from colour + the *actual* opacity curve (matching
            # _update_clim_fast's own transfer-function source) rather
            # than just cmap=..., which would default to fully opaque —
            # the colorbar swatch is supposed to visibly fade out at the
            # low-value end, same as the volume itself.
            from pyvista.plotting.colors import get_cmap_safe
            from pyvista.plotting.tools import opacity_transfer_function

            n = 256
            colors = (get_cmap_safe(self.cmap)(np.linspace(0.0, 1.0, n))[:, :3] * 255).astype(np.uint8)
            alphas = opacity_transfer_function(self.opacity, n).astype(np.uint8)
            rgba = np.column_stack([colors, alphas])
            mapper.lookup_table = pv.LookupTable(values=rgba)

        try:
            self._colorbar_actor = self.plotter.add_scalar_bar(
                # VTK's own built-in title for a *horizontal* scalar bar
                # overlaps the tick labels no matter how the height or
                # title/vertical-separation properties are tuned — draw
                # the title ourselves instead (see below), fully clear of
                # the labels.
                title="",
                mapper=self._volume_actor.mapper,
                vertical=False,
                position_x=cb_x,
                position_y=cb_y,
                width=cb_w,
                height=cb_h,
                color=text_color,
                font_family="times",
                label_font_size=self._label_font_size,
                outline=False,
                # VTK's own native labels always read the mapper's raw
                # scalar range — in log mode that range is log10(value),
                # so its numbers would show e.g. "-4.2" instead of the
                # real "6.3e-05". Suppressed entirely in favour of our
                # own labels (see _rebuild_colorbar_trim), which convert
                # back to real values for both modes.
                n_labels=0,
            )
            # DrawFrame's own outline wraps the *whole* actor (labels
            # included) — draw our own border/ticks around just the
            # gradient swatch instead (see _rebuild_colorbar_trim).
            self._colorbar_actor.SetDrawFrame(False)
            self._colorbar_actor.SetTextPad(12)
        except Exception:
            self._colorbar_actor = None
            return

        # Rendered as mathtext (see _render_mathtext_rgba/_build_mathtext_actor2d)
        # rather than a plain vtkTextActor, so a Quantity Units value
        # containing LaTeX-ish math (e.g. "$M_\\odot/h$") renders as
        # real math instead of literal dollar signs and backslashes.
        # Rebuilt from scratch each call (like the tick/label actors
        # above) since the title text/colour can change.
        if self._colorbar_title_actor is not None:
            self.plotter.renderer.RemoveActor2D(self._colorbar_title_actor)
            self._colorbar_title_actor = None

        if self.colorbar_title.strip():
            self._colorbar_title_actor = self._build_mathtext_actor2d(
                self.colorbar_title, self._theme_rgb(), self._label_font_size - 2, cb_x + cb_w / 2, cb_y + cb_h - 0.008
            )
            self.plotter.renderer.AddActor2D(self._colorbar_title_actor)

        self._rebuild_colorbar_trim(text_color)
        # add_scalar_bar()/_rebuild_colorbar_trim() above (re)create these
        # actors from scratch every time (called on every vmin/vmax/gamma/
        # colormap change), which would otherwise silently reset a
        # previously-toggled-off colorbar back to visible.
        self._colorbar_actor.SetVisibility(self.colorbar_visible)
        if self._colorbar_frame_actor is not None:
            self._colorbar_frame_actor.SetVisibility(self.colorbar_visible)
        if self._colorbar_tick_actor is not None:
            self._colorbar_tick_actor.SetVisibility(self.colorbar_visible)
        if self._colorbar_title_actor is not None:
            self._colorbar_title_actor.SetVisibility(self.colorbar_visible)
        for actor in self._colorbar_value_label_actors:
            actor.SetVisibility(self.colorbar_visible)

    def _rebuild_colorbar_trim(self, text_color):
        """Draw a thin border + small ticks around just the colorbar's
        gradient swatch, in display (pixel) coordinates queried from the
        actor itself once it's had a layout pass."""
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkLine, vtkPolyData
        from vtkmodules.vtkRenderingCore import vtkActor2D, vtkCoordinate, vtkPolyDataMapper2D

        renderer = self.plotter.renderer
        for old in (self._colorbar_frame_actor, self._colorbar_tick_actor, *self._colorbar_value_label_actors):
            if old is not None:
                try:
                    renderer.RemoveActor2D(old)
                except Exception:
                    pass
        self._colorbar_frame_actor = None
        self._colorbar_tick_actor = None
        self._colorbar_value_label_actors = []

        # GetScalarBarRect needs a layout pass to have happened already.
        renderer.GetRenderWindow().Render()
        rect = [0, 0, 0, 0]
        try:
            self._colorbar_actor.GetScalarBarRect(rect, renderer)
        except Exception:
            return
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            return

        coord = vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        coord.SetViewport(renderer)

        # Border: a closed rectangle exactly around the swatch.
        border_points = vtkPoints()
        for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h)):
            border_points.InsertNextPoint(px, py, 0.0)
        border_cells = vtkCellArray()
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            line = vtkLine()
            line.GetPointIds().SetId(0, a)
            line.GetPointIds().SetId(1, b)
            border_cells.InsertNextCell(line)
        border_poly = vtkPolyData()
        border_poly.SetPoints(border_points)
        border_poly.SetLines(border_cells)

        border_mapper = vtkPolyDataMapper2D()
        border_mapper.SetInputData(border_poly)
        border_mapper.SetTransformCoordinate(coord)
        self._colorbar_frame_actor = vtkActor2D()
        self._colorbar_frame_actor.SetMapper(border_mapper)
        self._colorbar_frame_actor.GetProperty().SetLineWidth(1)
        self._colorbar_frame_actor.GetProperty().SetColor(*self._theme_rgb())
        renderer.AddActor2D(self._colorbar_frame_actor)

        # Ticks: short marks at each label position, poking up from the
        # swatch's top edge (labels sit above, for a horizontal bar).
        tick_points = vtkPoints()
        tick_cells = vtkCellArray()
        tick_len = 6
        n = self._COLORBAR_N_LABELS
        for i in range(n):
            frac = i / (n - 1) if n > 1 else 0.5
            px = x + frac * w
            base = tick_points.InsertNextPoint(px, y + h, 0.0)
            tip = tick_points.InsertNextPoint(px, y + h + tick_len, 0.0)
            line = vtkLine()
            line.GetPointIds().SetId(0, base)
            line.GetPointIds().SetId(1, tip)
            tick_cells.InsertNextCell(line)
        tick_poly = vtkPolyData()
        tick_poly.SetPoints(tick_points)
        tick_poly.SetLines(tick_cells)

        tick_mapper = vtkPolyDataMapper2D()
        tick_mapper.SetInputData(tick_poly)
        tick_coord = vtkCoordinate()
        tick_coord.SetCoordinateSystemToDisplay()
        tick_coord.SetViewport(renderer)
        tick_mapper.SetTransformCoordinate(tick_coord)
        self._colorbar_tick_actor = vtkActor2D()
        self._colorbar_tick_actor.SetMapper(tick_mapper)
        self._colorbar_tick_actor.GetProperty().SetLineWidth(1)
        self._colorbar_tick_actor.GetProperty().SetColor(*self._theme_rgb())
        renderer.AddActor2D(self._colorbar_tick_actor)

        # Labels: drawn ourselves (rather than VTK's native scalar-bar
        # labels, which are suppressed via n_labels=0 in
        # _rebuild_colorbar) so a log-scale mapping can still show real
        # physical values here instead of raw log10 numbers.
        from vtkmodules.vtkRenderingCore import vtkTextActor

        vmin, vmax = self.current_clim
        for i in range(n):
            frac = i / (n - 1) if n > 1 else 0.5
            if self.value_scale == "log":
                lo = max(vmin, self._log_floor)
                hi = max(vmax, self._log_floor)
                if hi <= lo:
                    hi = lo * 10.0
                log_lo, log_hi = math.log10(lo), math.log10(hi)
                value = 10 ** (log_lo + frac * (log_hi - log_lo))
            else:
                value = vmin + frac * (vmax - vmin)

            label_actor = vtkTextActor()
            label_actor.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            label_actor.SetPosition(x + frac * w, y + h + tick_len + 2)
            label_actor.SetInput(f"{value:.2e}")
            prop = label_actor.GetTextProperty()
            prop.SetFontFamilyToTimes()
            prop.SetFontSize(self._label_font_size)
            prop.SetJustificationToCentered()
            prop.SetVerticalJustificationToBottom()
            prop.SetColor(*self._theme_rgb())
            prop.SetShadow(False)
            renderer.AddActor2D(label_actor)
            self._colorbar_value_label_actors.append(label_actor)

    def _theme_rgb(self):
        return (1.0, 1.0, 1.0) if self.is_dark_theme else (0.0, 0.0, 0.0)

    def redraw_moment0(self):
        if self.moment0_actor is not None:
            try:
                self.plotter.renderer.RemoveActor(self.moment0_actor)
            except Exception:
                pass

        nz = self.grid.dimensions[2]
        ny, nx = self.moment0.shape
        plane = pv.ImageData(dimensions=(nx, ny, 1), spacing=(1.0, 1.0, 1.0))
        plane.point_data["moment0"] = np.flipud(self.moment0).ravel(order="C")
        plane.translate((0.0, 0.0, (nz + 1.5) * self.vel_scale), inplace=True)

        fg_color = "white" if self.is_dark_theme else "black"
        self.moment0_actor = self.plotter.add_mesh(
            plane,
            scalars="moment0",
            cmap=self.cmap,
            clim=self._moment0_clim,
            opacity=0.85,
            show_edges=False,
            scalar_bar_args={
                "title": "Moment 0",
                "color": fg_color,
                "n_labels": 4,
                "position_x": 0.82,
                "position_y": 0.08,
                "width": 0.08,
                "height": 0.25,
            },
        )

    def redraw_sliders(self, text_color):
        self.plotter.clear_slider_widgets()

        def update_vmin(value):
            if value < self.current_clim[1]:
                self.current_clim[0] = value
                self.redraw_volume()

        def update_vmax(value):
            if value > self.current_clim[0]:
                self.current_clim[1] = value
                self.redraw_volume()

        self.plotter.add_slider_widget(
            update_vmin,
            rng=[self.d_min, self.d_max],
            value=self.current_clim[0],
            title="vmin (Noise Floor)",
            pointa=(0.05, 0.1),
            pointb=(0.35, 0.1),
            style="modern",
            color=text_color,
        )
        self.plotter.add_slider_widget(
            update_vmax,
            rng=[self.d_min, self.d_max],
            value=self.current_clim[1],
            title="vmax (Signal Peak)",
            pointa=(0.65, 0.1),
            pointb=(0.95, 0.1),
            style="modern",
            color=text_color,
        )

    def show(self, **kwargs):
        self.plotter.show(**kwargs)


def load_fits_cube(cube_path: str | Path):
    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]
    return cube


def load_cube_with_metadata(cube_path: str | Path, cube_index: int = 0):
    """Load a cube (FITS, HDF5, or a bare numpy array) plus whatever
    observational metadata its header/attrs expose. Returns ``(cube,
    info, extra)``:

    - ``info``: an ordered dict of only the display fields that were
      actually present — callers should skip any key that's missing
      rather than assume a fixed set.
    - ``extra``: ``{"axis_labels": (x, y, z), "spatial_scale":
      (value_per_voxel, unit) | None}`` for the viewer's axes/scale bar.

    ``cube_index`` only applies to a .npy/.npz file holding more than
    one volumetric cube (see numpy_cube_count) — ignored otherwise.
    """
    cube_path = Path(cube_path)
    if cube_path.suffix.lower() in (".h5", ".hdf5"):
        return _load_hdf5_cube_with_metadata(cube_path)
    if cube_path.suffix.lower() in (".npy", ".npz"):
        return _load_numpy_cube_with_metadata(cube_path, index=cube_index)
    return _load_fits_cube_with_metadata(cube_path)


def _raw_numpy_array(cube_path: Path):
    """The array as stored, before picking a single cube out of it — an
    .npz with more than one array in it, or any array with 4 dimensions
    (a stack of cubes along axis 0), holds more than one volumetric
    cube. Returns ``(array_or_None, keys_or_None)``: for a multi-array
    .npz, ``array_or_None`` is None and ``keys`` lists the array names
    (the caller picks one by index rather than by shape); otherwise
    ``keys`` is None and ``array_or_None`` is the loaded array."""
    if cube_path.suffix.lower() == ".npz":
        with np.load(cube_path) as data:
            keys = list(data.keys())
            if len(keys) > 1:
                return None, keys
            return np.asarray(data[keys[0]], dtype=np.float32), None
    return np.asarray(np.load(cube_path), dtype=np.float32), None


def numpy_cube_count(cube_path: Path) -> int:
    """How many separate volumetric cubes this .npy/.npz file holds —
    either multiple arrays in an .npz, or a single 4D array stacking
    several cubes along axis 0. 1 if it's just one plain cube."""
    array, keys = _raw_numpy_array(cube_path)
    if keys is not None:
        return len(keys)
    return array.shape[0] if array.ndim == 4 else 1


def _load_numpy_cube_with_metadata(cube_path: Path, index: int = 0):
    """A bare numpy array carries no header/attrs at all — ``info`` comes
    back empty and the GUI shows editable fields for the user to fill in
    instead of a read-only info card; ``extra`` uses generic axis labels
    and starts with no spatial scale (no scale bar) and no colorbar
    title (no quantity units) until the user supplies them. ``index``
    picks which cube out of a multi-array .npz or a 4D stack (see
    numpy_cube_count) — ignored for a file holding just one cube."""
    array, keys = _raw_numpy_array(cube_path)
    if keys is not None:
        with np.load(cube_path) as data:
            cube = np.asarray(data[keys[index]], dtype=np.float32)
    elif array.ndim == 4:
        cube = np.asarray(array[index], dtype=np.float32)
    else:
        cube = array
    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]

    info = {}
    extra = {
        "axis_labels": ("X", "Y", "Z"),
        "spatial_scale": None,
        "colorbar_title": "",
        "axis_ranges": None,
        "axis_label_formats": ("%.2f", "%.2f", "%.2f"),
        "axis_tick_formatters": (None, None, None),
        "axis_tick_units": (None, None, None),
    }
    return cube, info, extra


def _load_fits_cube_with_metadata(cube_path: Path):
    with fits.open(cube_path) as hdul:
        hdr = hdul[0].header
        cube = np.squeeze(np.asarray(hdul[0].data, dtype=np.float32))
        info = {}

        name = hdr.get("OBJECT")
        if name:
            info["Name"] = str(name).strip()

        telescope = hdr.get("TELESCOP") or hdr.get("INSTRUME")
        if telescope:
            info["Telescope/simulation"] = str(telescope).strip()

        # A real telescope cube always has RA/Dec + a spectral (velocity
        # or frequency) third axis, never a true third spatial dimension.
        info["Type of cube"] = "PPV"

        naxis1 = hdr.get("NAXIS1")
        naxis2 = hdr.get("NAXIS2")
        cdelt1 = hdr.get("CDELT1")
        if naxis1 and cdelt1:
            fov_arcsec = abs(float(cdelt1)) * 3600 * float(naxis1)
            info["Field of view"] = f'{fov_arcsec:.2f}"'

        # Absolute sky position of the field centre, in real sexagesimal.
        wcs2d = None
        if naxis1 and naxis2:
            try:
                wcs2d = WCS(hdr, naxis=2)
                center = wcs2d.pixel_to_world((naxis1 - 1) / 2.0, (naxis2 - 1) / 2.0)
                ra_str = center.ra.to_string(unit=u.hourangle, sep="hms", precision=1, pad=True)
                dec_str = center.dec.to_string(unit=u.deg, sep=("°", "'", '"'), precision=1, alwayssign=True, pad=True)
                info["Field center"] = f"{ra_str}, {dec_str}"
            except Exception:
                wcs2d = None

        bmaj = hdr.get("BMAJ")
        bmin = hdr.get("BMIN")
        if bmaj and bmin:
            info["Beam size"] = f'{float(bmaj) * 3600:.3f}" x {float(bmin) * 3600:.3f}"'

        cdelt3 = hdr.get("CDELT3")
        restfrq = hdr.get("RESTFRQ")
        cunit3 = str(hdr.get("CUNIT3", "")).strip().lower()
        spectral_is_velocity = bool(cdelt3 and restfrq and cunit3 in ("hz", ""))
        if cdelt3:
            if spectral_is_velocity:
                c_kms = 299792.458
                dv = abs(c_kms * float(cdelt3) / float(restfrq))
                info["Spectral resolution"] = f"{dv:.2f} km/s"
            else:
                info["Spectral resolution"] = f"{abs(float(cdelt3)):.4g} {cunit3 or 'Hz'}"

        zlabel = "km/s" if spectral_is_velocity else (cunit3 or "Hz")
        bunit = str(hdr.get("BUNIT", "")).strip()
        if bunit:
            info["Quantity Units"] = bunit
        extra = {
            "axis_labels": ("RA", "Dec", zlabel),
            "spatial_scale": None,
            "colorbar_title": bunit or "Intensity",
            "axis_ranges": None,
            "axis_label_formats": ("%.2f", "%.2f", "%.1f"),
            "axis_tick_formatters": (None, None, None),
            "axis_tick_units": (("h", "m", "s"), ("°", "'", '"'), None),
        }
        if naxis1 and cdelt1:
            extra["spatial_scale"] = (abs(float(cdelt1)) * 3600, "arcsec")
            extra["fov_value"] = fov_arcsec
            extra["fov_unit"] = "arcsec"
        if cdelt3:
            if spectral_is_velocity:
                extra["specres_value"] = dv
                extra["specres_unit"] = "km/s"
            else:
                extra["specres_value"] = abs(float(cdelt3))
                extra["specres_unit"] = cunit3 or "Hz"

        # Real-world tick values (arcsec offset from field centre for
        # RA/Dec, actual velocity/frequency for the spectral axis) rather
        # than raw voxel indices — assumes the field centre sits at the
        # midpoint of the array, which holds for the square-cropped/
        # rebinned cubes this viewer is meant for.
        cdelt2 = hdr.get("CDELT2") or cdelt1
        if naxis1 and cdelt1 and naxis2 and cdelt2:
            x_half = abs(float(cdelt1)) * 3600 * float(naxis1) / 2
            y_half = abs(float(cdelt2)) * 3600 * float(naxis2) / 2
            crpix3 = hdr.get("CRPIX3")
            crval3 = hdr.get("CRVAL3")
            if cdelt3 and crpix3 is not None and crval3 is not None:
                n_ch = cube.shape[0]
                freq0 = float(crval3) + float(cdelt3) * (0 - (float(crpix3) - 1))
                freq1 = float(crval3) + float(cdelt3) * ((n_ch - 1) - (float(crpix3) - 1))
                if spectral_is_velocity:
                    c_kms = 299792.458
                    z0 = c_kms * (float(restfrq) - freq0) / float(restfrq)
                    z1 = c_kms * (float(restfrq) - freq1) / float(restfrq)
                else:
                    z0, z1 = freq0, freq1
                extra["axis_ranges"] = ((-x_half, x_half), (-y_half, y_half), (z0, z1))
                extra["axis_label_formats"] = ("%.2f", "%.2f", "%.0f" if spectral_is_velocity else "%.3g")

            # Formatters (real sexagesimal RA/Dec via the field's actual
            # WCS, not a linear approximation) that the viewer's own
            # custom tick-label overlay calls with whatever real-world
            # tick values it decides to show (see
            # KinematicVolumeViewer._rebuild_custom_tick_labels) — this
            # keeps the conversion correct under any rotation/projection
            # while letting the overlay pick its own tick density/zoom
            # behaviour independent of it. Each returns component tuples
            # (sign, major, minor, seconds) rather than a final string, so
            # the overlay can abbreviate repeated leading components the
            # way astropy's own WCSAxes tick labels do.
            if wcs2d is not None:
                cx, cy = (naxis1 - 1) / 2.0, (naxis2 - 1) / 2.0
                cdelt1_arcsec = abs(float(cdelt1)) * 3600
                cdelt2_arcsec = abs(float(cdelt2)) * 3600

                def _ra_components(dx, _wcs=wcs2d, _cx=cx, _cy=cy, _scale=cdelt1_arcsec):
                    sky = _wcs.pixel_to_world(_cx + dx / _scale, _cy)
                    h, m, s = sky.ra.hms
                    return (1, int(round(h)), int(round(m)), float(s))

                def _dec_components(dy, _wcs=wcs2d, _cx=cx, _cy=cy, _scale=cdelt2_arcsec):
                    sky = _wcs.pixel_to_world(_cx, _cy + dy / _scale)
                    sign = -1 if sky.dec.deg < 0 else 1
                    d, m, s = sky.dec.dms
                    return (sign, int(round(abs(d))), int(round(abs(m))), float(abs(s)))

                # Plain-numeric velocity/frequency formatter, so the Z
                # axis renders through the same custom overlay as RA/Dec
                # instead of vtkCubeAxesActor's native labels.
                z_fmt = extra["axis_label_formats"][2]

                def _z_formatter(v, _f=z_fmt):
                    return _f % v

                extra["axis_tick_formatters"] = (_ra_components, _dec_components, _z_formatter)

    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]
    return cube, info, extra


def _load_hdf5_cube_with_metadata(cube_path: Path):
    import h5py

    with h5py.File(cube_path, "r") as f:
        for key in ("cube", "clean_cube", "noisy_cube"):
            if key in f:
                cube = np.asarray(f[key], dtype=np.float32)
                break
        else:
            raise ValueError(f"No recognised cube dataset found in {cube_path}")

        attrs = dict(f.attrs)

    info = {"Name": cube_path.stem, "Telescope/simulation": "Simulated cube"}

    fov_kpc = attrs.get("fov_kpc")
    if fov_kpc is not None:
        info["Field of view"] = f"{float(fov_kpc):.2f} kpc"

    spatial_res = attrs.get("spatial_resolution_kpc_per_px")
    if spatial_res is not None:
        info["Beam size"] = f"{float(spatial_res):.4g} kpc/px"

    spec_res = attrs.get("spectral_resolution_km_s")
    if spec_res is not None:
        info["Spectral resolution"] = f"{float(spec_res):.2f} km/s"

    # A velocity-binned mock spectral cube (PPV) has a spectral third
    # axis; without that, the third axis is a real spatial dimension —
    # a genuine 3D grid (PPP).
    info["Type of cube"] = "PPV" if spec_res is not None else "PPP"

    zlabel = "km/s" if spec_res is not None else "spectral units"
    extra = {
        "axis_labels": ("kpc", "kpc", zlabel),
        "spatial_scale": None,
        "colorbar_title": "Intensity",
        "axis_ranges": None,
        "axis_label_formats": ("%.2f", "%.2f", "%.1f"),
        "axis_tick_formatters": (None, None, None),
        "axis_tick_units": (None, None, None),
    }
    if spatial_res is not None:
        extra["spatial_scale"] = (float(spatial_res), "kpc")
    if fov_kpc is not None:
        extra["fov_value"] = float(fov_kpc)
        extra["fov_unit"] = "kpc"
    if spec_res is not None:
        extra["specres_value"] = float(spec_res)
        extra["specres_unit"] = "km/s"

    # Real-world tick values (kpc offset from the cube centre, and
    # velocity offset from systemic for the spectral axis) rather than
    # raw voxel indices.
    if spatial_res is not None:
        n_ch, ny, nx = cube.shape
        x_half = float(spatial_res) * nx / 2
        y_half = float(spatial_res) * ny / 2
        if spec_res is not None:
            z_half = float(spec_res) * n_ch / 2
            extra["axis_ranges"] = ((-x_half, x_half), (-y_half, y_half), (-z_half, z_half))
        else:
            extra["axis_ranges"] = ((-x_half, x_half), (-y_half, y_half), (0, n_ch - 1))

        # Plain-numeric formatters for all three axes, so the simulated
        # (kpc) case renders through the exact same custom tick/title
        # overlay as the observed (RA/Dec) case — same font, same
        # edge-following rotation, same zoom-adaptive density — rather
        # than mixing in vtkCubeAxesActor's native text for this case.
        x_fmt, y_fmt, z_fmt = extra["axis_label_formats"]
        extra["axis_tick_formatters"] = (
            lambda v, _f=x_fmt: _f % v,
            lambda v, _f=y_fmt: _f % v,
            lambda v, _f=z_fmt: _f % v,
        )

    if cube.ndim == 2:
        cube = cube[np.newaxis, :, :]
    return cube, info, extra
