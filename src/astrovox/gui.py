from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception as exc:  # pragma: no cover - runtime guard for builds without PyQt5
    QtCore = None
    QtGui = None
    QtWidgets = None
    QT_IMPORT_ERROR = exc
else:
    QT_IMPORT_ERROR = None

try:
    from pyvistaqt import QtInteractor
except Exception as exc:  # pragma: no cover - runtime guard for missing pyvistaqt
    QtInteractor = None
    PYVISTAQT_IMPORT_ERROR = exc
else:
    PYVISTAQT_IMPORT_ERROR = None

from astrovox.viewer import KinematicVolumeViewer, load_cube_with_metadata, numpy_cube_count


def _set_titlebar_theme(widget, is_dark: bool):
    """On macOS, force the native window titlebar to match the app's own
    light/dark theme (rather than always following the system-wide
    appearance) — e.g. the light-mode titlebar stays light even under
    macOS dark mode, the way apps like Songs do. Silently a no-op
    anywhere this isn't possible (non-macOS, or pyobjc unavailable)."""
    import sys

    if sys.platform != "darwin":
        return
    try:
        import objc
        from AppKit import NSAppearance
    except Exception:
        return
    try:
        view = objc.objc_object(c_void_p=int(widget.winId()))
        window = view.window()
        name = "NSAppearanceNameDarkAqua" if is_dark else "NSAppearanceNameAqua"
        window.setAppearance_(NSAppearance.appearanceNamed_(name))
    except Exception:
        pass


if QtInteractor is not None:

    class TrackpadInteractor(QtInteractor):
        """QtInteractor with native macOS trackpad gestures layered on top
        of PyVista's existing left-drag-to-rotate / scroll-to-zoom: a
        two-finger scroll pans the camera, a pinch (native zoom gesture)
        zooms, and a two-finger twist (native rotate gesture) spins the
        camera rig around the loaded cube's own centre (see set_pivot) —
        not around whatever the focal point drifted to after panning, and
        not around the viewport's centre, which is what a bare
        Camera.Roll() would do. Regular mouse wheels are untouched —
        trackpad scrolling is distinguished from a physical wheel by the
        presence of Qt's high-resolution ``pixelDelta``, which only
        trackpads (and Magic Mouse) populate."""

        _pivot = None

        def set_pivot(self, point):
            """Fix the world-space point that two-finger twist rotates
            around — call this once after loading a cube (e.g. with its
            bounds centre), so rotation stays anchored to the cube even
            after the view has been panned off-centre."""
            self._pivot = np.array(point, dtype=float)

        def wheelEvent(self, event):
            pixel_delta = event.pixelDelta()
            if not pixel_delta.isNull():
                self._pan_by_pixels(pixel_delta.x(), pixel_delta.y())
                event.accept()
                return
            super().wheelEvent(event)

        def event(self, event):
            if event.type() == QtCore.QEvent.NativeGesture:
                gesture_type = event.gestureType()
                if gesture_type == QtCore.Qt.ZoomNativeGesture:
                    self._zoom_by_factor(1.0 + event.value())
                    return True
                if gesture_type == QtCore.Qt.RotateNativeGesture:
                    self._roll_by_degrees(event.value())
                    return True
            return super().event(event)

        def _pan_by_pixels(self, dx_px, dy_px):
            renderer = self.renderer
            camera = renderer.GetActiveCamera()
            size = self.GetRenderWindow().GetSize()
            if size[1] == 0:
                return

            fp = np.array(camera.GetFocalPoint())
            pos = np.array(camera.GetPosition())
            view_up = np.array(camera.GetViewUp())
            direction = fp - pos
            distance = np.linalg.norm(direction)
            if distance == 0:
                return
            view_dir = direction / distance
            right = np.cross(view_dir, view_up)
            right_norm = np.linalg.norm(right)
            if right_norm == 0:
                return
            right = right / right_norm
            true_up = np.cross(right, view_dir)

            # World-space size of one pixel at the focal plane, so the
            # dragged point under the fingers stays under the fingers.
            if camera.GetParallelProjection():
                world_height = camera.GetParallelScale() * 2
            else:
                angle = np.radians(camera.GetViewAngle())
                world_height = 2 * distance * np.tan(angle / 2)
            scale = world_height / size[1]

            delta = (-dx_px * scale) * right + (dy_px * scale) * true_up
            camera.SetPosition(*(pos + delta))
            camera.SetFocalPoint(*(fp + delta))
            renderer.ResetCameraClippingRange()
            self.render()

        def _zoom_by_factor(self, factor):
            if factor <= 0:
                return
            renderer = self.renderer
            renderer.GetActiveCamera().Dolly(factor)
            renderer.ResetCameraClippingRange()
            self.render()

        _suppress_free_revert = False

        def snap_to_axis_plane(self, plane):
            """Jump to a straight-on view of the given coordinate plane
            ("X-Y", "Y-Z", or "X-Z"), looking along its normal axis,
            centred on the cube's pivot and keeping the current zoom
            (distance)."""
            renderer = self.renderer
            camera = renderer.GetActiveCamera()
            pivot = self._pivot if self._pivot is not None else np.array(camera.GetFocalPoint())
            distance = camera.GetDistance()

            directions = {
                "X-Y": (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
                "Y-Z": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
                "X-Z": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            }
            direction, up = directions[plane]

            # The camera changes below fire the same ModifiedEvent that
            # manual rotate/pan/zoom does (which is what reverts the pill
            # selector to "Free") — suppress that reaction for this
            # programmatic, snap-requested change only.
            self._suppress_free_revert = True
            camera.SetPosition(*(pivot + direction * distance))
            camera.SetFocalPoint(*pivot)
            camera.SetViewUp(*up)
            renderer.ResetCameraClippingRange()
            self.render()
            self._suppress_free_revert = False

        def _roll_by_degrees(self, degrees):
            # Confirmed inverted on real hardware: rotation needs to spin
            # opposite to the finger twist direction Qt reports.
            degrees = -degrees
            renderer = self.renderer
            camera = renderer.GetActiveCamera()
            pos = np.array(camera.GetPosition())
            fp = np.array(camera.GetFocalPoint())
            up = np.array(camera.GetViewUp())

            # Rotate the whole camera rig (position, focal point, and
            # view-up) around the axis through the fixed pivot, parallel
            # to the current view direction. Camera.Roll() alone rotates
            # about that axis only when it already passes through the
            # pivot; once panning has moved the focal point away from the
            # cube's centre, Roll() instead spins the image around the
            # viewport's centre. Rotating relative to the pivot keeps the
            # cube visually anchored regardless of how far it's been
            # panned off-centre or how far away the camera is.
            pivot = self._pivot if self._pivot is not None else fp
            axis = fp - pos
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                return
            axis = axis / axis_norm

            theta = np.radians(degrees)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            def _rotate(v):
                return v * cos_t + np.cross(axis, v) * sin_t + axis * np.dot(axis, v) * (1 - cos_t)

            new_pos = pivot + _rotate(pos - pivot)
            new_fp = pivot + _rotate(fp - pivot)
            new_up = _rotate(up)

            camera.SetPosition(*new_pos)
            camera.SetFocalPoint(*new_fp)
            camera.SetViewUp(*new_up)
            renderer.ResetCameraClippingRange()
            self.render()
else:
    TrackpadInteractor = None

CUBE_FILE_FILTER = (
    "FITS/HDF5/NumPy cubes (*.fits *.fit *.fts *.h5 *.hdf5 *.npy *.npz);;All files (*)"
)
_CUBE_EXTENSIONS = {".fits", ".fit", ".fts", ".h5", ".hdf5", ".npy", ".npz"}
_NUMPY_EXTENSIONS = {".npy", ".npz"}

# Colour palette shared by every custom control below — sliders, pills,
# the theme button — so the whole column reads as one consistent style.
_THEMES = {
    "dark": dict(
        BG="#0a0a0a", CARD_BG="#111111", TEXT="#999999", ACCENT="#b39ddb",
        ENTRY_BG="#1a1a1a", SLIDER_BORDER="#4a3b66", SLIDER_TROUGH="#111111",
        SLIDER_THUMBHOV="#d1c4f0", PILL_NOR="#1e1e1e", PILL_HOV="#2e2140",
        PILL_SEL_FG="#000000", CARD_TEXT="#ffffff", DANGER="#e57373",
    ),
    "light": dict(
        BG="#f0ede6", CARD_BG="#ffffff", TEXT="#444444", ACCENT="#7e57c2",
        ENTRY_BG="#ffffff", SLIDER_BORDER="#c5b3e6", SLIDER_TROUGH="#ffffff",
        SLIDER_THUMBHOV="#d8c6f2", PILL_NOR="#dedad0", PILL_HOV="#ece0f7",
        PILL_SEL_FG="#ffffff", CARD_TEXT="#1a1a1a", DANGER="#c62828",
    ),
}

# Opacity transfer functions for the Linear / Log / Power scale selector.
# All three are explicit arrays derived from the same [0, 1] ramp.
#
# IMPORTANT: PyVista's opacity_transfer_function() only rescales a custom
# array to the [0, 255] byte range it needs internally when the array is
# SHORTER than the LUT's colour count; when it's exactly equal (as any
# 256-sample array is, matching the default 256-colour LUT), it takes the
# array as already-scaled uint8 and passes it through untouched. Handing
# it raw floats in [0, 1] in that case makes every opacity value truncate
# to ~0/255 — i.e. an almost fully transparent volume — which is exactly
# why Log/Power silently rendered blank. Pre-scaling to uint8 ourselves
# sidesteps that branch entirely.
_N_OPACITY_SAMPLES = 256
_T = np.linspace(0.0, 1.0, _N_OPACITY_SAMPLES)
_DEFAULT_GAMMA = 3.0


def _to_uint8_opacity(curve):
    return np.clip(np.asarray(curve) * 255.0, 0, 255).astype(np.uint8)


def _linear_opacity():
    return _to_uint8_opacity(_T)


def _log_opacity():
    return _to_uint8_opacity(np.log1p(9 * _T) / np.log1p(9))


def _power_opacity(gamma: float):
    return _to_uint8_opacity(_T ** gamma)


# Colormaps chosen so the volume blends into the background: dark-mode
# maps run toward black at the low end, light-mode maps run toward white.
_DARK_COLORMAPS = ["plasma", "inferno", "magma", "viridis", "cividis"]
_LIGHT_COLORMAPS = ["cubehelix_r", "gray_r", "bone_r", "pink_r", "afmhot_r"]


class SquareViewportContainer(QtWidgets.QWidget if QtWidgets is not None else object):
    """Hosts a single child widget, always sized/centered as a square
    inscribed in the available space, regardless of window resizing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._child = None

    def set_child(self, widget):
        if self._child is not None:
            self._child.setParent(None)
        self._child = widget
        widget.setParent(self)
        self._relayout()
        widget.show()

    def clear_child(self):
        # Reparent the current child out immediately, rather than leaving
        # that to the next set_child() call — the caller is about to
        # deleteLater() this widget, and set_child() would then be
        # reaching into an already-deleted C/C++ object once Qt gets
        # around to processing that deferred deletion.
        if self._child is not None:
            self._child.setParent(None)
            self._child = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self):
        if self._child is None:
            return
        side = max(0, min(self.width(), self.height()))
        # Flush to the top-left edge rather than centred, so the viewer
        # starts right at the window's edge instead of floating in the
        # middle of a wider-than-tall container.
        self._child.setGeometry(0, 0, side, side)


class LabeledSlider(QtWidgets.QWidget if QtWidgets is not None else object):
    """Slider + editable value box: a filled accent pill-bar slider (no
    separate knob) with a bordered entry box beside it. Rather than a
    plain heading above the slider, it's labelled with a small rich-text
    math symbol inline to the slider's left — e.g. V_min, V_max, gamma;
    ``symbol_html`` here is that label, as Qt rich text (e.g.
    ``"V<sub>min</sub>"``)."""

    _STEPS = 1000

    if QtCore is not None:
        valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, symbol_html, minimum, maximum, value, fmt="{:.4g}", parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._fmt = fmt
        self._updating = False
        self._log = False

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._symbol = QtWidgets.QLabel(self)
        self._symbol.setTextFormat(QtCore.Qt.RichText)
        self._symbol.setText(symbol_html)
        self._symbol.setFixedWidth(30)
        row.addWidget(self._symbol)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self._slider.setFixedHeight(18)
        self._slider.setMinimum(0)
        self._slider.setMaximum(self._STEPS)
        self._slider.setValue(self._to_step(value))
        row.addWidget(self._slider, 1)

        self._entry = QtWidgets.QLineEdit(fmt.format(value), self)
        self._entry.setFixedWidth(56)
        self._entry.setFixedHeight(18)
        self._entry.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(self._entry)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._entry.editingFinished.connect(self._on_entry_committed)

    def _log_bounds(self):
        """A positive (lo, hi) pair to log-interpolate across — real
        vmin/vmax can legitimately be 0 or negative (astronomical data's
        noise floor), which log10 can't represent, so the slider's low
        end is clamped to a tiny fraction of the high end instead."""
        hi = self._max if self._max > 0 else 1e-300
        lo = self._min if self._min > 0 else hi * 1e-6
        if lo <= 0:
            lo = 1e-300
        return lo, hi

    def _to_step(self, v):
        if self._max <= self._min:
            return 0
        if self._log:
            lo, hi = self._log_bounds()
            v = max(lo, min(hi, v))
            frac = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) if hi > lo else 0.0
        else:
            frac = (v - self._min) / (self._max - self._min)
        return int(round(max(0.0, min(1.0, frac)) * self._STEPS))

    def _to_value(self, step):
        if self._log:
            lo, hi = self._log_bounds()
            frac = step / self._STEPS
            return 10 ** (math.log10(lo) + frac * (math.log10(hi) - math.log10(lo)))
        return self._min + (step / self._STEPS) * (self._max - self._min)

    def set_log_scale(self, enabled: bool):
        """Switch the slider's position<->value mapping between linear
        and logarithmic — the stored/emitted value always stays in real
        (linear) units, only how mouse position along the track maps to
        that value changes. Without this, a slider spanning many orders
        of magnitude (typical for astronomical intensity data) is
        unusable in Log mode: almost the entire track corresponds to a
        sliver of the value range."""
        if enabled == self._log:
            return
        current = self.value()
        self._log = enabled
        self._updating = True
        self._slider.setValue(self._to_step(current))
        self._updating = False

    def _on_slider_changed(self, step):
        if self._updating:
            return
        self._updating = True
        v = self._to_value(step)
        self._entry.setText(self._fmt.format(v))
        self._updating = False
        self.valueChanged.emit(v)

    def _on_entry_committed(self):
        if self._updating:
            return
        try:
            v = float(self._entry.text())
        except ValueError:
            return
        v = max(self._min, min(self._max, v))
        self._updating = True
        self._slider.setValue(self._to_step(v))
        self._entry.setText(self._fmt.format(v))
        self._updating = False
        self.valueChanged.emit(v)

    def set_range(self, minimum, maximum, value):
        self._min, self._max = minimum, maximum
        self._updating = True
        self._slider.setValue(self._to_step(value))
        self._entry.setText(self._fmt.format(value))
        self._updating = False

    def value(self):
        return self._to_value(self._slider.value())

    def set_enabled_dimmed(self, enabled: bool):
        """Enable/disable this slider and visually dim it (low opacity)
        when disabled — used for the Power gamma slider, which only
        matters while 'Power' is the selected scale."""
        self._slider.setEnabled(enabled)
        self._entry.setEnabled(enabled)
        effect = self.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(self)
            self.setGraphicsEffect(effect)
        effect.setOpacity(1.0 if enabled else 0.35)

    def apply_theme(self, palette):
        self._symbol.setStyleSheet(
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 13px; font-style: italic; }}"
        )
        self._entry.setStyleSheet(f"""
            QLineEdit {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 3px;
                padding: 1px 4px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
        """)
        # Filled pill-bar look: a solid accent-coloured fill from the left up to the current
        # value, on a dark trough, with a thin accent border around the
        # whole bar and no separate visible thumb/knob.
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                background: {palette['SLIDER_TROUGH']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 4px;
                height: 18px;
            }}
            QSlider::sub-page:horizontal {{
                background: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 4px;
            }}
            QSlider::add-page:horizontal {{
                background: {palette['SLIDER_TROUGH']};
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {palette['ACCENT']};
                width: 2px;
                margin: 0px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {palette['SLIDER_THUMBHOV']};
            }}
        """)


class PillSelector(QtWidgets.QWidget if QtWidgets is not None else object):
    """Segmented pill selector: a row of flat rectangular buttons, the
    selected one filled with the accent colour."""

    if QtCore is not None:
        valueChanged = QtCore.pyqtSignal(str)

    def __init__(self, values, selected=None, parent=None, pill_height=24, pill_width=None):
        super().__init__(parent)
        self._values = list(values)
        self._buttons = {}

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)

        for val in self._values:
            btn = QtWidgets.QPushButton(val, self)
            btn.setCheckable(True)
            btn.setFixedHeight(pill_height)
            if pill_width is not None:
                btn.setFixedWidth(pill_width)
            btn.clicked.connect(lambda _checked, v=val: self.valueChanged.emit(v))
            self._group.addButton(btn)
            self._buttons[val] = btn
            layout.addWidget(btn)

        selected = selected or (self._values[0] if self._values else None)
        if selected in self._buttons:
            self._buttons[selected].setChecked(True)

    def current_value(self):
        checked = self._group.checkedButton()
        return checked.text() if checked is not None else None

    def set_selected(self, value):
        """Change the selected pill programmatically, without emitting
        valueChanged — used to revert to "Free" when the camera moves for
        a reason other than clicking a snap pill (e.g. manual rotate/pan/
        zoom), which must not re-trigger a snap."""
        btn = self._buttons.get(value)
        if btn is not None:
            btn.setChecked(True)
        self.refresh_style()

    def refresh_style(self):
        """Force a QSS re-polish on every pill. A *programmatic* setChecked
        (unlike an actual click) doesn't reliably repaint the ":checked"
        pseudo-state — most notably right after construction, before the
        widget has ever been shown, which is why the pills can look jumbled
        on first launch until something (e.g. a theme toggle) forces a
        repaint. Call this once the page holding the pills is actually
        visible to fix that up front."""
        for b in self._buttons.values():
            b.style().unpolish(b)
            b.style().polish(b)
            b.update()

    def apply_theme(self, palette):
        for btn in self._buttons.values():
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {palette['PILL_NOR']};
                    color: {palette['CARD_TEXT']};
                    border: none;
                    border-radius: 2px;
                    font-family: Georgia, 'Times New Roman';
                    font-size: 10px;
                    font-weight: bold;
                    padding: 2px 10px;
                }}
                QPushButton:hover:!checked {{
                    background: {palette['PILL_HOV']};
                }}
                QPushButton:checked {{
                    background: {palette['ACCENT']};
                    color: {palette['PILL_SEL_FG']};
                }}
            """)


class ToggleGrid(QtWidgets.QWidget if QtWidgets is not None else object):
    """A 2-column grid of independently checkable pills — unlike
    PillSelector, these are plain on/off toggles (no exclusive
    group), used for the Visual aesthetics section."""

    if QtCore is not None:
        toggled = QtCore.pyqtSignal(str, bool)

    def __init__(self, labels_defaults, parent=None):
        super().__init__(parent)
        self._buttons = {}

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, (label, default) in enumerate(labels_defaults):
            btn = QtWidgets.QPushButton(label, self)
            btn.setCheckable(True)
            btn.setChecked(default)
            btn.setFixedHeight(22)
            btn.clicked.connect(lambda checked, name=label: self.toggled.emit(name, checked))
            self._buttons[label] = btn
            grid.addWidget(btn, i // 2, i % 2)

    def is_checked(self, label):
        btn = self._buttons.get(label)
        return btn.isChecked() if btn is not None else False

    def column_width(self) -> int:
        """Actual on-screen width of one grid column (i.e. one pill) —
        used by CubeOutlineRow to make its own toggle button match."""
        if self._buttons:
            return next(iter(self._buttons.values())).width()
        return 0

    def pill_height(self) -> int:
        """The fixed height every pill in this grid uses — used by
        CubeOutlineRow to make its own toggle button match."""
        if self._buttons:
            return next(iter(self._buttons.values())).height()
        return 0

    def set_pill_enabled(self, label, enabled: bool):
        """Disable/dim (or restore) a single pill — used to gate the
        Scalebar pill for numpy cubes until a valid Field of view has
        been entered."""
        btn = self._buttons.get(label)
        if btn is None:
            return
        btn.setEnabled(enabled)
        effect = btn.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(btn)
            btn.setGraphicsEffect(effect)
        effect.setOpacity(1.0 if enabled else 0.35)

    def set_checked_silent(self, label, checked: bool):
        """Change a pill's checked state without emitting `toggled` —
        the caller is expected to separately drive whatever the pill
        controls, since a programmatic setChecked() doesn't fire the
        button's own clicked signal."""
        btn = self._buttons.get(label)
        if btn is not None:
            btn.setChecked(checked)

    def apply_theme(self, palette):
        for btn in self._buttons.values():
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: {palette['PILL_NOR']};
                    color: {palette['CARD_TEXT']};
                    border: none;
                    border-radius: 2px;
                    font-family: Georgia, 'Times New Roman';
                    font-size: 10px;
                    font-weight: bold;
                    padding: 4px 4px;
                }}
                QPushButton:hover:!checked {{
                    background: {palette['PILL_HOV']};
                }}
                QPushButton:checked {{
                    background: {palette['ACCENT']};
                    color: {palette['PILL_SEL_FG']};
                }}
            """)


class CubeOutlineRow(QtWidgets.QWidget if QtWidgets is not None else object):
    """"Cube Outline" toggle plus its own Thickness/Style dropdowns — a
    standalone row (rather than a ToggleGrid entry, since it needs room
    for two extra dropdowns) in the Visual Aesthetics card. The toggle
    sits in the same grid-column-0 as ToggleGrid's pills (same width),
    with the two dropdowns sharing column 1. Both dropdowns show a
    drawn line sample as their icon (no text) so the thickness/style is
    visible directly in the closed combo box, and dim/become unclickable
    whenever the outline itself is switched off."""

    if QtCore is not None:
        toggled = QtCore.pyqtSignal(bool)
        thicknessChanged = QtCore.pyqtSignal(int)
        styleChanged = QtCore.pyqtSignal(str)

    _THICKNESSES = (1, 2, 3, 4, 5)
    _STYLES = ("solid", "dashed", "dotted")
    _ICON_SIZE = QtCore.QSize(34, 14) if QtCore is not None else None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._arrow_paths = {}
        self._icon_cache = {}

        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self._toggle_btn = QtWidgets.QPushButton("Cube Outline", self)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.setFixedHeight(18)  # overridden by set_toggle_height() once themed
        self._toggle_btn.clicked.connect(self._on_toggle_clicked)
        grid.addWidget(self._toggle_btn, 0, 0)

        combos_widget = QtWidgets.QWidget(self)
        combos_row = QtWidgets.QHBoxLayout(combos_widget)
        combos_row.setContentsMargins(0, 0, 0, 0)
        combos_row.setSpacing(6)

        self._thickness_combo = QtWidgets.QComboBox(self)
        self._thickness_combo.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        self._thickness_combo.setIconSize(self._ICON_SIZE)
        self._thickness_combo.setFixedWidth(66)
        self._thickness_combo.setFixedHeight(20)
        for t in self._THICKNESSES:
            self._thickness_combo.addItem("", t)
        self._thickness_combo.setCurrentIndex(self._THICKNESSES.index(1))
        self._thickness_combo.currentIndexChanged.connect(
            lambda i: self.thicknessChanged.emit(self._thickness_combo.itemData(i))
        )
        combos_row.addWidget(self._thickness_combo)

        self._style_combo = QtWidgets.QComboBox(self)
        self._style_combo.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        self._style_combo.setIconSize(self._ICON_SIZE)
        self._style_combo.setFixedWidth(66)
        self._style_combo.setFixedHeight(20)
        for s in self._STYLES:
            self._style_combo.addItem("", s)
        self._style_combo.currentIndexChanged.connect(
            lambda i: self.styleChanged.emit(self._style_combo.itemData(i))
        )
        combos_row.addWidget(self._style_combo)

        grid.addWidget(combos_widget, 0, 1)

    def set_toggle_width(self, width: int):
        """Match the toggle pill's width to ToggleGrid's actual rendered
        column width (its own grid, sized independently, doesn't land on
        the same pixel width by construction since its second column
        holds two comboboxes instead of a single pill)."""
        if width > 0:
            self._toggle_btn.setFixedWidth(width)

    def set_toggle_height(self, height: int):
        """Match the toggle pill's height to ToggleGrid's own pills, so
        "Cube Outline" doesn't stand out as a different size."""
        if height > 0:
            self._toggle_btn.setFixedHeight(height)

    def _on_toggle_clicked(self, checked):
        self._set_dropdowns_enabled(checked)
        self.toggled.emit(checked)

    def is_checked(self) -> bool:
        return self._toggle_btn.isChecked()

    def thickness(self) -> int:
        return self._thickness_combo.currentData()

    def line_style(self) -> str:
        return self._style_combo.currentData()

    def _set_dropdowns_enabled(self, enabled: bool):
        for combo in (self._thickness_combo, self._style_combo):
            combo.setEnabled(enabled)
            effect = combo.graphicsEffect()
            if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
                effect = QtWidgets.QGraphicsOpacityEffect(combo)
                combo.setGraphicsEffect(effect)
            effect.setOpacity(1.0 if enabled else 0.35)

    def _arrow_icon_path(self, color_hex):
        path = self._arrow_paths.get(color_hex)
        if path is not None:
            return path
        size = 12
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(color_hex))
        triangle = QtGui.QPolygonF([
            QtCore.QPointF(size * 0.22, size * 0.38),
            QtCore.QPointF(size * 0.78, size * 0.38),
            QtCore.QPointF(size * 0.50, size * 0.68),
        ])
        painter.drawPolygon(triangle)
        painter.end()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pixmap.save(tmp.name, "PNG")
        self._arrow_paths[color_hex] = tmp.name
        return tmp.name

    def _line_sample_icon(self, kind, value, color_hex):
        key = (kind, value, color_hex)
        icon = self._icon_cache.get(key)
        if icon is not None:
            return icon
        w, h = self._ICON_SIZE.width(), self._ICON_SIZE.height()
        pixmap = QtGui.QPixmap(w, h)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor(color_hex))
        if kind == "thickness":
            pen.setWidthF(float(value))
            pen.setCapStyle(QtCore.Qt.RoundCap)
        else:
            pen.setWidthF(2.2)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setStyle({
                "solid": QtCore.Qt.SolidLine,
                "dashed": QtCore.Qt.DashLine,
                "dotted": QtCore.Qt.DotLine,
            }[value])
        painter.setPen(pen)
        painter.drawLine(4, h // 2, w - 4, h // 2)
        painter.end()
        icon = QtGui.QIcon(pixmap)
        self._icon_cache[key] = icon
        return icon

    def apply_theme(self, palette):
        self._toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background: {palette['PILL_NOR']};
                color: {palette['CARD_TEXT']};
                border: none;
                border-radius: 2px;
                font-family: Georgia, 'Times New Roman';
                font-size: 10px;
                font-weight: bold;
                padding: 4px 4px;
            }}
            QPushButton:hover:!checked {{
                background: {palette['PILL_HOV']};
            }}
            QPushButton:checked {{
                background: {palette['ACCENT']};
                color: {palette['PILL_SEL_FG']};
            }}
        """)
        combo_css = f"""
            QComboBox {{
                background: {palette['ENTRY_BG']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 6px;
                padding: 3px 6px;
            }}
            QComboBox:hover {{
                border: 1px solid {palette['ACCENT']};
            }}
            QComboBox:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 16px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: url({self._arrow_icon_path(palette['ACCENT'])});
                width: 8px;
                height: 8px;
                margin-right: 2px;
            }}
            QComboBox QAbstractItemView {{
                background: {palette['ENTRY_BG']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 6px;
                padding: 4px;
                outline: none;
                selection-background-color: {palette['ACCENT']};
            }}
        """
        self._thickness_combo.setStyleSheet(combo_css)
        self._style_combo.setStyleSheet(combo_css)

        accent = palette["ACCENT"]
        prev_thickness = self.thickness()
        self._thickness_combo.blockSignals(True)
        for i, t in enumerate(self._THICKNESSES):
            self._thickness_combo.setItemIcon(i, self._line_sample_icon("thickness", t, accent))
        self._thickness_combo.setCurrentIndex(self._THICKNESSES.index(prev_thickness))
        self._thickness_combo.blockSignals(False)

        prev_style = self.line_style()
        self._style_combo.blockSignals(True)
        for i, s in enumerate(self._STYLES):
            self._style_combo.setItemIcon(i, self._line_sample_icon("style", s, accent))
        self._style_combo.setCurrentIndex(self._STYLES.index(prev_style))
        self._style_combo.blockSignals(False)

        self._set_dropdowns_enabled(self._toggle_btn.isChecked())


class PlaybackRow(QtWidgets.QWidget if QtWidgets is not None else object):
    """One "<label>: [Play/Pause] − [speed] +" row: a square Play/Pause
    button (accent-filled while playing, plain pill otherwise) plus a
    −/+ stepper flanking an
    editable speed readout (small pill buttons, accent-coloured symbols).
    Used for the Animation section's Azimuth/Elevation auto-rotate rows —
    this widget only exposes play state and speed; CubeViewerApp owns the
    actual camera-rotation timer."""

    if QtCore is not None:
        toggled = QtCore.pyqtSignal(bool)
        speedChanged = QtCore.pyqtSignal(float)

    def __init__(self, label_text, speed=30.0, speed_min=-180.0, speed_max=180.0, speed_step=2.0, parent=None):
        super().__init__(parent)
        self._playing = False
        self._speed = speed
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._speed_step = speed_step

        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._play_btn = QtWidgets.QPushButton("▶", self)
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedSize(24, 24)
        self._play_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._play_btn.clicked.connect(self._on_play_clicked)
        row.addWidget(self._play_btn)

        self._label = QtWidgets.QLabel(label_text, self)
        row.addWidget(self._label)
        row.addStretch(1)

        self._minus_btn = QtWidgets.QPushButton("−", self)
        self._minus_btn.setFixedSize(20, 24)
        self._minus_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._minus_btn.clicked.connect(lambda: self._step_speed(-self._speed_step))
        row.addWidget(self._minus_btn)

        self._speed_edit = QtWidgets.QLineEdit(f"{speed:g}", self)
        self._speed_edit.setFixedWidth(40)
        self._speed_edit.setFixedHeight(24)
        self._speed_edit.setAlignment(QtCore.Qt.AlignCenter)
        self._speed_edit.editingFinished.connect(self._on_speed_typed)
        row.addWidget(self._speed_edit)

        self._plus_btn = QtWidgets.QPushButton("+", self)
        self._plus_btn.setFixedSize(20, 24)
        self._plus_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._plus_btn.clicked.connect(lambda: self._step_speed(self._speed_step))
        row.addWidget(self._plus_btn)

    def _on_play_clicked(self):
        self._playing = self._play_btn.isChecked()
        self._play_btn.setText("⏸" if self._playing else "▶")
        self.toggled.emit(self._playing)

    def _step_speed(self, delta):
        self._set_speed(self._speed + delta)

    def _on_speed_typed(self):
        try:
            v = float(self._speed_edit.text())
        except ValueError:
            v = self._speed
        self._set_speed(v)

    def _set_speed(self, v):
        self._speed = max(self._speed_min, min(self._speed_max, v))
        self._speed_edit.setText(f"{self._speed:g}")
        self.speedChanged.emit(self._speed)

    def speed(self):
        return self._speed

    def is_playing(self):
        return self._playing

    def stop(self):
        """Stop playback programmatically (e.g. when resetting back to
        the upload page) — mirrors an actual click on the Play/Pause
        button so CubeViewerApp's own toggled-state bookkeeping (the
        animation timer, Record/Save gating) stays in sync."""
        if self._playing:
            self._play_btn.setChecked(False)
            self._on_play_clicked()

    def apply_theme(self, palette):
        self._label.setStyleSheet(
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; }}"
        )
        self._play_btn.setStyleSheet(f"""
            QPushButton {{
                background: {palette['PILL_NOR']};
                color: {palette['CARD_TEXT']};
                border: none;
                border-radius: 3px;
                font-size: 11px;
            }}
            QPushButton:hover:!checked {{
                background: {palette['PILL_HOV']};
            }}
            QPushButton:checked {{
                background: {palette['ACCENT']};
                color: {palette['PILL_SEL_FG']};
            }}
        """)
        step_btn_css = f"""
            QPushButton {{
                background: {palette['PILL_NOR']};
                color: {palette['ACCENT']};
                border: none;
                border-radius: 3px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {palette['PILL_HOV']};
            }}
        """
        self._minus_btn.setStyleSheet(step_btn_css)
        self._plus_btn.setStyleSheet(step_btn_css)
        self._speed_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 3px;
                padding: 1px 2px;
                font-family: Georgia, 'Times New Roman';
                font-size: 10px;
            }}
            QLineEdit:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
        """)


class RecordControl(QtWidgets.QWidget if QtWidgets is not None else object):
    """"Record Video" button that walks through record → 3-2-1 countdown →
    recording → stopped states. This widget owns only the UI/animation
    (countdown ticks, swapping which controls are shown) and emits
    signals at the points that need real action (starting the countdown,
    actually starting/stopping frame capture, saving, resetting) —
    CubeViewerApp owns the actual viewport recording via the signals
    below.

    Row layout, left to right: [accent box] [Save .mp4] [FPS: input].
    The accent box itself doubles as Stop once recording starts (no
    separate Stop control) — "⏺ Record Video" while idle, "⏺ <n>" during
    the countdown, a plain "⏹" while recording (click anywhere on it to
    stop), and "Reset" (no icon) once stopped, which discards the clip
    and returns to idle. "Save .mp4" appears (disabled) from the
    countdown onward and only actually becomes clickable once stopped —
    it's the one thing that keeps the recording. The accent box spans the
    full row while idle; Save and the FPS field are fixed-width, so its
    stretch factor shrinks it to make room for them rather than the row
    growing wider. The FPS field is only editable while idle — the value
    is locked in for the whole clip once recording starts."""

    if QtCore is not None:
        recordClicked = QtCore.pyqtSignal()
        countdownFinished = QtCore.pyqtSignal()
        stopClicked = QtCore.pyqtSignal()
        saveClicked = QtCore.pyqtSignal()
        resetClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "idle"  # idle | countdown | recording | stopped
        self._gate_enabled = True
        self._countdown_value = 0
        self._fps = 30
        self._fps_min, self._fps_max = 1, 60

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._accent_box = QtWidgets.QWidget(self)
        self._accent_box.setFixedHeight(28)
        self._accent_box.setCursor(QtCore.Qt.PointingHandCursor)
        self._accent_box.mousePressEvent = self._on_accent_box_pressed
        self._box_row = QtWidgets.QHBoxLayout(self._accent_box)
        self._box_row.setContentsMargins(10, 0, 10, 0)
        self._box_row.setSpacing(6)

        self._indicator = QtWidgets.QLabel("⏺", self._accent_box)
        self._indicator.setFixedSize(20, 20)
        self._indicator.setAlignment(QtCore.Qt.AlignCenter)

        self._text_label = QtWidgets.QLabel("Record Video", self._accent_box)

        self._countdown_label = QtWidgets.QLabel("", self._accent_box)
        self._countdown_label.setFixedWidth(16)
        self._countdown_label.setAlignment(QtCore.Qt.AlignCenter)

        outer.addWidget(self._accent_box, 1)

        self._save_btn = QtWidgets.QPushButton("Save .mp4", self)
        self._save_btn.setFixedHeight(28)
        self._save_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._save_btn.clicked.connect(self._on_save_clicked)
        outer.addWidget(self._save_btn)

        self._fps_label = QtWidgets.QLabel("FPS:", self)
        outer.addWidget(self._fps_label)
        self._fps_minus_btn = QtWidgets.QPushButton("−", self)
        self._fps_minus_btn.setFixedSize(20, 24)
        self._fps_minus_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._fps_minus_btn.clicked.connect(lambda: self._step_fps(-1))
        outer.addWidget(self._fps_minus_btn)
        # Same width as PlaybackRow's speed entry, for visual
        # consistency between the two "type a number" fields in this
        # column.
        self._fps_edit = QtWidgets.QLineEdit(str(self._fps), self)
        self._fps_edit.setFixedWidth(40)
        self._fps_edit.setFixedHeight(24)
        self._fps_edit.setAlignment(QtCore.Qt.AlignCenter)
        self._fps_edit.editingFinished.connect(self._on_fps_typed)
        outer.addWidget(self._fps_edit)
        self._fps_plus_btn = QtWidgets.QPushButton("+", self)
        self._fps_plus_btn.setFixedSize(20, 24)
        self._fps_plus_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._fps_plus_btn.clicked.connect(lambda: self._step_fps(1))
        outer.addWidget(self._fps_plus_btn)

        self._countdown_timer = QtCore.QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._on_countdown_tick)

        self._rebuild_row()

    def _on_accent_box_pressed(self, event):
        if self._state == "idle":
            self.recordClicked.emit()
        elif self._state == "recording":
            self._on_stop_clicked()
        elif self._state == "stopped":
            self._state = "idle"
            self._rebuild_row()
            self.resetClicked.emit()

    def fps(self) -> int:
        return self._fps

    def _step_fps(self, delta):
        self._set_fps(self._fps + delta)

    def _on_fps_typed(self):
        try:
            v = int(float(self._fps_edit.text()))
        except ValueError:
            v = self._fps
        self._set_fps(v)

    def _set_fps(self, v):
        self._fps = max(self._fps_min, min(self._fps_max, int(v)))
        self._fps_edit.setText(str(self._fps))

    def start_countdown(self, seconds: int = 3):
        self._state = "countdown"
        self._countdown_value = seconds
        self._countdown_label.setText(str(seconds))
        self._countdown_timer.start()
        self._rebuild_row()

    def _on_countdown_tick(self):
        self._countdown_value -= 1
        if self._countdown_value <= 0:
            self._countdown_timer.stop()
            self.countdownFinished.emit()
        else:
            self._countdown_label.setText(str(self._countdown_value))

    def enter_recording(self):
        self._state = "recording"
        self._rebuild_row()

    def _on_stop_clicked(self):
        if self._state != "recording":
            return
        self._state = "stopped"
        self._rebuild_row()
        self.stopClicked.emit()

    def _on_save_clicked(self):
        if self._state != "stopped":
            return
        self.saveClicked.emit()

    def reset_idle(self):
        self._countdown_timer.stop()
        self._state = "idle"
        self._rebuild_row()

    def set_animation_gate(self, enabled: bool):
        """Recording only makes sense while the cube is actually moving —
        the whole row (record/stop button, Save .mp4, FPS field) is
        disabled and dimmed whenever both animation rows are paused, and
        active again once at least one of them is playing.

        This dims every widget in the row individually rather than
        stacking a second QGraphicsOpacityEffect on the container: Qt's
        software effect compositing breaks (some widgets, particularly
        Fusion-styled ones, silently stop painting at all) when an
        effect-bearing widget is nested inside another effect-bearing
        ancestor."""
        self._gate_enabled = enabled
        self._rebuild_row()

    @staticmethod
    def _set_dimmed(widget, enabled: bool):
        widget.setEnabled(enabled)
        effect = widget.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        effect.setOpacity(1.0 if enabled else 0.35)

    def _rebuild_row(self):
        while self._box_row.count():
            self._box_row.takeAt(0)
        for w in (self._indicator, self._text_label, self._countdown_label):
            w.hide()

        gate = self._gate_enabled
        self._set_dimmed(self._accent_box, gate)

        fps_enabled = gate and self._state == "idle"
        self._set_dimmed(self._fps_edit, fps_enabled)
        self._set_dimmed(self._fps_label, fps_enabled)
        self._set_dimmed(self._fps_minus_btn, fps_enabled)
        self._set_dimmed(self._fps_plus_btn, fps_enabled)

        if self._state == "idle":
            self._box_row.addStretch(1)
            self._box_row.addWidget(self._indicator)
            self._indicator.setText("⏺")
            self._box_row.addWidget(self._text_label)
            self._text_label.setText("Record Video")
            self._box_row.addStretch(1)
            self._indicator.show()
            self._text_label.show()
            self._save_btn.hide()
        elif self._state == "countdown":
            self._box_row.addStretch(1)
            self._box_row.addWidget(self._indicator)
            self._indicator.setText("⏺")
            self._box_row.addWidget(self._countdown_label)
            self._box_row.addStretch(1)
            self._indicator.show()
            self._countdown_label.show()
            self._save_btn.show()
            self._set_dimmed(self._save_btn, False)
        elif self._state == "recording":
            self._box_row.addStretch(1)
            self._box_row.addWidget(self._indicator)
            self._indicator.setText("⏹")
            self._box_row.addStretch(1)
            self._indicator.show()
            self._save_btn.show()
            self._set_dimmed(self._save_btn, False)
        else:  # "stopped"
            self._box_row.addStretch(1)
            self._box_row.addWidget(self._text_label)
            self._text_label.setText("Reset")
            self._box_row.addStretch(1)
            self._text_label.show()
            self._save_btn.show()
            self._set_dimmed(self._save_btn, gate)

    def apply_theme(self, palette):
        # Contrasting text/icon colour against the accent fill — the same
        # convention as a selected pill (PILL_SEL_FG): black on dark mode's
        # lighter pastel accent, white on light mode's more saturated one.
        fg = palette["PILL_SEL_FG"]
        self._accent_box.setStyleSheet(f"""
            QWidget {{
                background: {palette['ACCENT']};
                border-radius: 4px;
            }}
        """)
        self._indicator.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                background: transparent;
                border: 1px solid {fg};
                border-radius: 3px;
                padding: 1px;
                font-size: 11px;
            }}
        """)
        self._text_label.setStyleSheet(
            f"QLabel {{ color: {fg}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; font-weight: bold; }}"
        )
        self._countdown_label.setStyleSheet(
            f"QLabel {{ color: {fg}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 13px; font-weight: bold; }}"
        )
        self._save_btn.setStyleSheet(f"""
            QPushButton {{
                background: {palette['PILL_NOR']};
                color: {palette['ACCENT']};
                border: none;
                border-radius: 4px;
                padding: 0 10px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover:enabled {{
                background: {palette['PILL_HOV']};
            }}
        """)
        self._fps_label.setStyleSheet(
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; }}"
        )
        self._fps_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 3px;
                padding: 1px 2px;
                font-family: Georgia, 'Times New Roman';
                font-size: 10px;
            }}
            QLineEdit:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
        """)
        step_btn_css = f"""
            QPushButton {{
                background: {palette['PILL_NOR']};
                color: {palette['ACCENT']};
                border: none;
                border-radius: 3px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover:enabled {{
                background: {palette['PILL_HOV']};
            }}
        """
        self._fps_minus_btn.setStyleSheet(step_btn_css)
        self._fps_plus_btn.setStyleSheet(step_btn_css)


class StaticFrameControl(QtWidgets.QWidget if QtWidgets is not None else object):
    """"Save Static Frame" button — a single-shot counterpart to
    RecordControl's video recording: click the accent box to
    immediately grab the current viewport frame (in whatever format the
    locked-in "Format:" dropdown says), then click "Save .<ext>" to write
    it to disk via a save dialog. Unlike video, there's no countdown or
    stop step — capture is instant. Once captured, the accent box turns
    into a plain "Reset" button (no icon), matching RecordControl's
    stopped-state Reset: clicking it just clears back to the initial
    "ready to capture" state rather than capturing again."""

    if QtCore is not None:
        captureClicked = QtCore.pyqtSignal()
        saveClicked = QtCore.pyqtSignal(str)
        resetClicked = QtCore.pyqtSignal()

    _FORMATS = ("png", "jpg", "pdf", "tiff")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "idle"  # "idle" or "captured"
        self._gate_enabled = True
        self._arrow_paths = {}

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        self._accent_box = QtWidgets.QWidget(self)
        self._accent_box.setFixedHeight(28)
        self._accent_box.setCursor(QtCore.Qt.PointingHandCursor)
        self._accent_box.mousePressEvent = self._on_accent_box_pressed
        self._box_row = QtWidgets.QHBoxLayout(self._accent_box)
        self._box_row.setContentsMargins(10, 0, 10, 0)
        self._box_row.setSpacing(6)

        # A crop/viewfinder-style "four corners" glyph — the closest
        # single Unicode symbol to the requested corner-bracket icon.
        self._icon = QtWidgets.QLabel("⛶", self._accent_box)
        self._icon.setFixedSize(20, 20)
        self._icon.setAlignment(QtCore.Qt.AlignCenter)

        self._text_label = QtWidgets.QLabel("Capture Static Frame", self._accent_box)

        self._box_row.addStretch(1)
        self._box_row.addWidget(self._icon)
        self._box_row.addWidget(self._text_label)
        self._box_row.addStretch(1)

        outer.addWidget(self._accent_box, 1)

        self._save_btn = QtWidgets.QPushButton(f"Save .{self._FORMATS[0]}", self)
        self._save_btn.setFixedHeight(28)
        self._save_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._save_btn.hide()
        outer.addWidget(self._save_btn)

        self._format_label = QtWidgets.QLabel("Format:", self)
        outer.addWidget(self._format_label)

        self._format_combo = QtWidgets.QComboBox(self)
        self._format_combo.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        self._format_combo.addItems(self._FORMATS)
        self._format_combo.currentTextChanged.connect(self._on_format_changed)
        outer.addWidget(self._format_combo)

    def _on_accent_box_pressed(self, event):
        if self._state == "idle":
            self._state = "captured"
            self._rebuild_row()
            self._save_btn.setText(f"Save .{self._format_combo.currentText()}")
            self.captureClicked.emit()
        else:  # "captured" -> Reset, back to idle (does not re-capture)
            self._state = "idle"
            self._rebuild_row()
            self.resetClicked.emit()

    def _on_format_changed(self, text):
        if self._state == "idle":
            self._save_btn.setText(f"Save .{text}")

    def _on_save_clicked(self):
        if self._state != "captured":
            return
        self.saveClicked.emit(self._format_combo.currentText())

    def format(self) -> str:
        return self._format_combo.currentText()

    def reset(self):
        """Back to idle — used when a new cube is loaded, so a stale
        frame from the previous cube can't be saved under it."""
        self._state = "idle"
        self._rebuild_row()

    def _rebuild_row(self):
        gate = self._gate_enabled
        self._set_dimmed(self._accent_box, gate)
        self._set_dimmed(self._format_combo, gate and self._state == "idle")
        self._set_dimmed(self._format_label, gate and self._state == "idle")
        if self._state == "idle":
            self._icon.show()
            self._text_label.setText("Capture Static Frame")
            self._save_btn.hide()
        else:  # "captured"
            self._icon.hide()
            self._text_label.setText("Reset")
            self._save_btn.show()
            self._set_dimmed(self._save_btn, gate)

    def set_animation_gate(self, enabled: bool):
        """The static-frame capture only makes sense while the cube is
        actually still — disabled/dimmed whenever either animation row is
        playing, active again once both are paused.

        Dims every widget in the row individually rather than stacking a
        second QGraphicsOpacityEffect on the container: Qt's software
        effect compositing breaks (some widgets, particularly
        Fusion-styled ones, silently stop painting at all) when an
        effect-bearing widget is nested inside another effect-bearing
        ancestor."""
        self._gate_enabled = enabled
        self._rebuild_row()

    @staticmethod
    def _set_dimmed(widget, enabled: bool):
        widget.setEnabled(enabled)
        effect = widget.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(effect)
        effect.setOpacity(1.0 if enabled else 0.35)

    def _arrow_icon_path(self, color_hex):
        path = self._arrow_paths.get(color_hex)
        if path is not None:
            return path
        size = 12
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(color_hex))
        triangle = QtGui.QPolygonF([
            QtCore.QPointF(size * 0.22, size * 0.38),
            QtCore.QPointF(size * 0.78, size * 0.38),
            QtCore.QPointF(size * 0.50, size * 0.68),
        ])
        painter.drawPolygon(triangle)
        painter.end()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pixmap.save(tmp.name, "PNG")
        self._arrow_paths[color_hex] = tmp.name
        return tmp.name

    def apply_theme(self, palette):
        fg = palette["PILL_SEL_FG"]
        self._accent_box.setStyleSheet(f"""
            QWidget {{
                background: {palette['ACCENT']};
                border-radius: 4px;
            }}
        """)
        self._icon.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                background: transparent;
                border: 1px solid {fg};
                border-radius: 3px;
                padding: 1px;
                font-size: 11px;
            }}
        """)
        self._text_label.setStyleSheet(
            f"QLabel {{ color: {fg}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; font-weight: bold; }}"
        )
        self._format_label.setStyleSheet(
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; }}"
        )
        self._format_combo.setStyleSheet(f"""
            QComboBox {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 6px;
                padding: 3px 8px;
                font-family: Georgia, 'Times New Roman';
                font-size: 10px;
            }}
            QComboBox:hover {{
                border: 1px solid {palette['ACCENT']};
            }}
            QComboBox:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 18px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: url({self._arrow_icon_path(palette['ACCENT'])});
                width: 8px;
                height: 8px;
                margin-right: 3px;
            }}
            QComboBox QAbstractItemView {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 6px;
                padding: 4px;
                outline: none;
                selection-background-color: {palette['ACCENT']};
                selection-color: {palette['PILL_SEL_FG']};
            }}
        """)
        self._save_btn.setStyleSheet(f"""
            QPushButton {{
                background: {palette['PILL_NOR']};
                color: {palette['ACCENT']};
                border: none;
                border-radius: 4px;
                padding: 0 10px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover:enabled {{
                background: {palette['PILL_HOV']};
            }}
        """)


class ColormapSelector(QtWidgets.QWidget if QtWidgets is not None else object):
    """Labelled colormap dropdown, styled like the rest of this app's
    controls. Offers a different curated list of colormaps per theme (dark
    maps that fade to black, light maps that fade to white), remembering
    the last choice made in each theme independently."""

    if QtCore is not None:
        valueChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = {"dark": _DARK_COLORMAPS[0], "light": _LIGHT_COLORMAPS[0]}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._label = QtWidgets.QLabel("Colormap", self)
        layout.addWidget(self._label)

        self._combo = QtWidgets.QComboBox(self)
        # macOS's native combobox style largely ignores QSS on the
        # ::drop-down/::down-arrow subcontrols (it paints its own native
        # arrow glyph regardless), which is why a custom arrow drawn via
        # stylesheet rendered as a stray artifact. Fusion is a
        # QSS-driven style that actually honours these rules.
        self._combo.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo)

        self._theme_name = "dark"
        self._arrow_paths = {}  # colour hex -> temp PNG path (see _arrow_icon_path)

    def _arrow_icon_path(self, color_hex):
        # Qt's ::down-arrow subcontrol only honours an actual `image:`
        # asset — border-triangle CSS tricks silently fall back to
        # Fusion's stock glyph on this build. Painting a tiny chevron to
        # a cached temp PNG per colour sidesteps that entirely.
        path = self._arrow_paths.get(color_hex)
        if path is not None:
            return path

        size = 12
        pixmap = QtGui.QPixmap(size, size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(color_hex))
        triangle = QtGui.QPolygonF([
            QtCore.QPointF(size * 0.22, size * 0.38),
            QtCore.QPointF(size * 0.78, size * 0.38),
            QtCore.QPointF(size * 0.50, size * 0.68),
        ])
        painter.drawPolygon(triangle)
        painter.end()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pixmap.save(tmp.name, "PNG")
        self._arrow_paths[color_hex] = tmp.name
        return tmp.name

    def _on_combo_changed(self, text):
        if not text:
            return
        self._current[self._theme_name] = text
        self.valueChanged.emit(text)

    def set_theme_maps(self, theme_name: str):
        """Switch the dropdown's option list to the given theme's curated
        colormaps, restoring that theme's last selection."""
        self._theme_name = theme_name
        maps = _DARK_COLORMAPS if theme_name == "dark" else _LIGHT_COLORMAPS
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(maps)
        selected = self._current.get(theme_name, maps[0])
        if selected not in maps:
            selected = maps[0]
        self._combo.setCurrentText(selected)
        self._combo.blockSignals(False)
        self._current[theme_name] = selected

    def current_value(self):
        return self._combo.currentText()

    def apply_theme(self, palette):
        self._label.setStyleSheet(
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; }}"
        )
        # A softer resting border (only the accent colour on hover/focus)
        # plus a hand-drawn triangle arrow read more like a modern web
        # <select> than a stock OS combobox.
        self._combo.setStyleSheet(f"""
            QComboBox {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 7px;
                padding: 5px 10px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
            }}
            QComboBox:hover {{
                border: 1px solid {palette['ACCENT']};
            }}
            QComboBox:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 24px;
                border: none;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: url({self._arrow_icon_path(palette['ACCENT'])});
                width: 10px;
                height: 10px;
                margin-right: 4px;
            }}
            QComboBox QAbstractItemView {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 7px;
                padding: 4px;
                outline: none;
                selection-background-color: {palette['ACCENT']};
                selection-color: {palette['PILL_SEL_FG']};
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 22px;
                padding: 2px 6px;
                border-radius: 4px;
            }}
        """)


class ThemeButton(QtWidgets.QPushButton if QtWidgets is not None else object):
    """Theme toggle button: bordered square showing a sun/moon glyph,
    colours inverting on hover."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 30)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def apply_theme(self, palette, is_dark: bool):
        self.setText("☀" if is_dark else "☾")  # ☀ / ☾
        self.setStyleSheet(f"""
            QPushButton {{
                background: {palette['BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                font-family: Georgia, 'Times New Roman';
                font-size: 15px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {palette['ACCENT']};
                color: {palette['BG']};
            }}
        """)


class ResetButton(QtWidgets.QPushButton if QtWidgets is not None else object):
    """Bottom-left "Reset" button, same row as the theme toggle — pastel
    red (rather than accent-coloured) since it's a destructive action:
    discards the current cube and returns to the upload/drop-zone page,
    after a confirmation dialog (see CubeViewerApp._on_reset_clicked)."""

    def __init__(self, parent=None):
        super().__init__("Reset", parent)
        self.setFixedHeight(30)
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def apply_theme(self, palette):
        # Qt's own sizeHint() (which determines how wide the button
        # actually gets) is computed from self.font(), not from the QSS
        # font-family below — setting a real QFont too (matching, so
        # both agree) is what avoids the button being sized too narrow
        # for its own text and clipping it.
        font = QtGui.QFont("Courier New", 10, QtGui.QFont.Bold)
        self.setFont(font)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {palette['BG']};
                color: {palette['DANGER']};
                border: 1px solid {palette['DANGER']};
                border-radius: 3px;
                padding: 4px 30px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {palette['DANGER']};
                color: {palette['BG']};
            }}
        """)


class LinkButton(QtWidgets.QPushButton if QtWidgets is not None else object):
    """Bottom-row button that opens an external URL — same accent
    styling as the theme toggle (border colour, hover-inverts), used for
    "Docs" and "GitHub" alongside Reset and the theme button."""

    def __init__(self, text, url, parent=None):
        super().__init__(text, parent)
        self._url = url
        self.setFixedHeight(30)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.clicked.connect(self._open)

    def _open(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._url))

    def apply_theme(self, palette):
        # See ResetButton.apply_theme for why the QFont is set explicitly
        # alongside the matching QSS font-family — sizeHint() needs it.
        self.setFont(QtGui.QFont("Courier New", 10, QtGui.QFont.Bold))
        self.setStyleSheet(f"""
            QPushButton {{
                background: {palette['BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 3px;
                padding: 4px 30px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {palette['ACCENT']};
                color: {palette['BG']};
            }}
        """)


class Card(QtWidgets.QFrame if QtWidgets is not None else object):
    """Bordered card wrapper: a thin accent-bordered outline around a
    padded content area."""

    def __init__(self, parent=None, faint_border=False):
        super().__init__(parent)
        self._faint_border = faint_border
        self._inner_layout = QtWidgets.QVBoxLayout(self)
        self._inner_layout.setContentsMargins(12, 10, 12, 10)
        self._inner_layout.setSpacing(6)

    def layout_for_content(self):
        return self._inner_layout

    def apply_theme(self, palette):
        border = "rgba(128, 128, 128, 0.25)" if self._faint_border else palette['SLIDER_BORDER']
        self.setStyleSheet(f"""
            Card {{
                background: {palette['CARD_BG']};
                border: 1px solid {border};
                border-radius: 4px;
            }}
        """)


class ManualInfoForm(Card if QtWidgets is not None else object):
    """Editable metadata form shown for every loaded cube — pre-filled
    from whatever a FITS/HDF5 header actually provided (see prefill()),
    or left blank/"$$" for a numpy array, which carries no header/attrs
    at all. Field of view feeds the scale bar and Quantity Units feeds
    the colorbar title once filled in; Spectral Resolution only unlocks
    once the cube type indicates a velocity axis (anything but PPP).
    For a numpy cube, Field of view/Spectral Resolution also feed the
    axis tick labels/titles directly (see
    KinematicVolumeViewer.set_manual_axis_scale) — for a FITS/HDF5 cube,
    those two fields stay purely informational/editable, since its own
    RA/Dec/velocity (or kpc) axis system is already correct."""

    if QtCore is not None:
        fovChanged = QtCore.pyqtSignal(object, str)  # (x, y, z) each float | None
        specResChanged = QtCore.pyqtSignal(object, str)  # value: float | None
        quantityUnitsChanged = QtCore.pyqtSignal(str)
        cubeTypeChanged = QtCore.pyqtSignal(str)

    _CUBE_TYPES = ("PPP", "PPV", "PVP", "VPP")

    def __init__(self, parent=None):
        super().__init__(parent)
        content = self.layout_for_content()
        self._field_labels = []

        self._name_edit = QtWidgets.QLineEdit(self)
        content.addLayout(self._inline_row(self._add_label(content, "Name:"), self._name_edit))

        self._telescope_edit = QtWidgets.QLineEdit(self)
        content.addLayout(self._inline_row(self._add_label(content, "Telescope/simulation:"), self._telescope_edit))

        self._cube_type_selector = PillSelector(
            list(self._CUBE_TYPES), selected="PPP", parent=self, pill_height=20
        )
        self._cube_type_selector.valueChanged.connect(self._on_cube_type_changed)
        content.addLayout(self._inline_row(self._add_label(content, "Type of cube:"), self._cube_type_selector))

        fov_row = QtWidgets.QHBoxLayout()
        fov_row.setSpacing(4)
        self._fov_label = QtWidgets.QLabel("Field of view:", self)
        fov_row.addWidget(self._fov_label)
        self._fov_x_edit = QtWidgets.QLineEdit(self)
        self._fov_x_edit.setFixedWidth(40)
        self._fov_x_edit.setValidator(QtGui.QDoubleValidator())
        fov_row.addWidget(self._fov_x_edit)
        self._fov_times1_label = QtWidgets.QLabel("×", self)
        fov_row.addWidget(self._fov_times1_label)
        self._fov_y_edit = QtWidgets.QLineEdit(self)
        self._fov_y_edit.setFixedWidth(40)
        self._fov_y_edit.setValidator(QtGui.QDoubleValidator())
        fov_row.addWidget(self._fov_y_edit)
        self._fov_times2_label = QtWidgets.QLabel("×", self)
        fov_row.addWidget(self._fov_times2_label)
        self._fov_z_edit = QtWidgets.QLineEdit(self)
        self._fov_z_edit.setFixedWidth(40)
        self._fov_z_edit.setValidator(QtGui.QDoubleValidator())
        fov_row.addWidget(self._fov_z_edit)
        self._fov_unit_label = QtWidgets.QLabel("Unit:", self)
        fov_row.addWidget(self._fov_unit_label)
        self._fov_unit_edit = QtWidgets.QLineEdit(self)
        self._fov_unit_edit.setFixedWidth(50)
        self._fov_unit_edit.setText("$$")
        fov_row.addWidget(self._fov_unit_edit)
        fov_row.addStretch(1)
        content.addLayout(fov_row)
        for w in (self._fov_x_edit, self._fov_y_edit, self._fov_z_edit, self._fov_unit_edit):
            w.textChanged.connect(self._on_fov_changed)

        specres_row = QtWidgets.QHBoxLayout()
        specres_row.setSpacing(6)
        self._specres_label = QtWidgets.QLabel("Spectral Resolution:", self)
        specres_row.addWidget(self._specres_label)
        self._specres_value_edit = QtWidgets.QLineEdit(self)
        self._specres_value_edit.setFixedWidth(60)
        self._specres_value_edit.setValidator(QtGui.QDoubleValidator())
        specres_row.addWidget(self._specres_value_edit)
        self._specres_unit_label = QtWidgets.QLabel("Unit:", self)
        specres_row.addWidget(self._specres_unit_label)
        self._specres_unit_edit = QtWidgets.QLineEdit(self)
        self._specres_unit_edit.setFixedWidth(60)
        self._specres_unit_edit.setText("$$")
        specres_row.addWidget(self._specres_unit_edit)
        specres_row.addStretch(1)
        content.addLayout(specres_row)
        self._specres_value_edit.textChanged.connect(self._on_specres_changed)
        self._specres_unit_edit.textChanged.connect(self._on_specres_changed)

        self._quantity_edit = QtWidgets.QLineEdit(self)
        self._quantity_edit.setText("$$")
        content.addLayout(self._inline_row(self._add_label(content, "Quantity Units:"), self._quantity_edit))
        self._quantity_edit.textChanged.connect(lambda t: self.quantityUnitsChanged.emit(t.strip()))

        self._set_specres_enabled(False)
        self._set_fov_z_enabled(True)  # default cube type is PPP

    def _add_label(self, layout, text):
        lbl = QtWidgets.QLabel(text, self)
        self._field_labels.append(lbl)
        return lbl

    @staticmethod
    def _inline_row(label, edit):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(label)
        row.addWidget(edit, 1)
        return row

    def _on_cube_type_changed(self, value):
        self._set_specres_enabled(value != "PPP")
        # A non-PPP cube has only two real spatial (Field of view)
        # dimensions — the third is the velocity axis, governed by
        # Spectral Resolution instead.
        self._set_fov_z_enabled(value == "PPP")
        self.cubeTypeChanged.emit(value)

    @staticmethod
    def _set_dimmed_widgets(widgets, enabled: bool):
        for w in widgets:
            w.setEnabled(enabled)
            effect = w.graphicsEffect()
            if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
                effect = QtWidgets.QGraphicsOpacityEffect(w)
                w.setGraphicsEffect(effect)
            effect.setOpacity(1.0 if enabled else 0.35)

    def _set_specres_enabled(self, enabled: bool):
        self._set_dimmed_widgets(
            (self._specres_label, self._specres_value_edit, self._specres_unit_label, self._specres_unit_edit),
            enabled,
        )

    def _set_fov_z_enabled(self, enabled: bool):
        self._set_dimmed_widgets((self._fov_times2_label, self._fov_z_edit), enabled)

    def _on_fov_changed(self):
        def parse(edit):
            text = edit.text().strip()
            try:
                return float(text) if text else None
            except ValueError:
                return None

        unit = self._fov_unit_edit.text().strip()
        x, y, z = parse(self._fov_x_edit), parse(self._fov_y_edit), parse(self._fov_z_edit)
        self.fovChanged.emit((x, y, z), unit)

    def _on_specres_changed(self):
        text = self._specres_value_edit.text().strip()
        unit = self._specres_unit_edit.text().strip()
        try:
            value = float(text) if text else None
        except ValueError:
            value = None
        self.specResChanged.emit(value, unit)

    def name(self) -> str:
        return self._name_edit.text().strip()

    def telescope(self) -> str:
        return self._telescope_edit.text().strip()

    def cube_type(self) -> str:
        return self._cube_type_selector.current_value()

    def prefill(self, info: dict, extra: dict):
        """Populate every field from whatever metadata a FITS/HDF5
        header actually provided — the same editable textbox UI as a
        numpy cube, just pre-filled instead of starting blank. Anything
        the header didn't supply is simply left at its default (blank,
        or "$$" for a unit field)."""
        if info.get("Name"):
            self._name_edit.setText(info["Name"])
        if info.get("Telescope/simulation"):
            self._telescope_edit.setText(info["Telescope/simulation"])

        cube_type = info.get("Type of cube")
        if cube_type in self._CUBE_TYPES:
            self._cube_type_selector.set_selected(cube_type)
            self._cube_type_selector.refresh_style()
            self._set_specres_enabled(cube_type != "PPP")
            self._set_fov_z_enabled(cube_type == "PPP")

        if "fov_value" in extra:
            # FITS/HDF5 only ever provide one Field of view value (the
            # RA/Dec plane's side length) — applied to both X and Y;
            # there's no separate depth-axis FOV to fill Z with (its own
            # axis is either velocity, already correct on its own, or
            # for a PPP HDF5 cube simply not something these loaders
            # report).
            self._fov_x_edit.setText(f"{extra['fov_value']:.4g}")
            self._fov_y_edit.setText(f"{extra['fov_value']:.4g}")
        if extra.get("fov_unit"):
            self._fov_unit_edit.setText(extra["fov_unit"])
        if "specres_value" in extra:
            self._specres_value_edit.setText(f"{extra['specres_value']:.4g}")
        if extra.get("specres_unit"):
            self._specres_unit_edit.setText(extra["specres_unit"])
        if info.get("Quantity Units"):
            self._quantity_edit.setText(info["Quantity Units"])

    def reset(self, default_name: str = ""):
        """Back to blank — called each time a new numpy cube is loaded so
        stale values from a previous one can't linger. Name is
        pre-filled with the file's own name as a convenient default,
        rather than starting empty."""
        self._name_edit.setText(default_name)
        self._telescope_edit.clear()
        self._cube_type_selector.set_selected("PPP")
        self._cube_type_selector.refresh_style()
        self._set_specres_enabled(False)
        self._set_fov_z_enabled(True)
        self._fov_x_edit.clear()
        self._fov_y_edit.clear()
        self._fov_z_edit.clear()
        self._fov_unit_edit.setText("$$")
        self._specres_value_edit.clear()
        self._specres_unit_edit.setText("$$")
        self._quantity_edit.setText("$$")

    def apply_theme(self, palette):
        super().apply_theme(palette)
        for lbl in self._field_labels + [
            self._fov_label, self._fov_times1_label, self._fov_times2_label, self._fov_unit_label,
            self._specres_label, self._specres_unit_label,
        ]:
            lbl.setStyleSheet(
                f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
                f"font-family: Georgia, 'Times New Roman'; font-size: 11px; }}"
            )
        edit_css = f"""
            QLineEdit {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['SLIDER_BORDER']};
                border-radius: 3px;
                padding: 2px 6px;
                font-family: Georgia, 'Times New Roman';
                font-size: 11px;
            }}
            QLineEdit:focus {{
                border: 1px solid {palette['SLIDER_THUMBHOV']};
            }}
        """
        for edit in (
            self._name_edit, self._telescope_edit,
            self._fov_x_edit, self._fov_y_edit, self._fov_z_edit, self._fov_unit_edit,
            self._specres_value_edit, self._specres_unit_edit, self._quantity_edit,
        ):
            edit.setStyleSheet(edit_css)
        self._cube_type_selector.apply_theme(palette)


class PlusIcon(QtWidgets.QWidget if QtWidgets is not None else object):
    """A '+' drawn directly (two crossed rounded bars) rather than a font
    glyph — a QLabel's "+" text is centred by Qt using the font's full
    ascent/descent box, which for most fonts isn't visually symmetric
    around the glyph itself, leaving the character looking off-centre no
    matter how its container is aligned. Painting it ourselves guarantees
    it sits exactly in the middle of this widget's own rect."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QtGui.QColor(255, 255, 255, 40)

    def set_color(self, color: QtGui.QColor):
        self._color = color
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(self._color)
        w, h = self.width(), self.height()
        thickness = min(w, h) * 0.16
        length = min(w, h) * 0.85
        cx, cy = w / 2.0, h / 2.0
        radius = thickness * 0.15
        painter.drawRoundedRect(
            QtCore.QRectF(cx - thickness / 2, cy - length / 2, thickness, length), radius, radius
        )
        painter.drawRoundedRect(
            QtCore.QRectF(cx - length / 2, cy - thickness / 2, length, thickness), radius, radius
        )


class DropZone(QtWidgets.QWidget if QtWidgets is not None else object):
    """Blank landing page: a low-opacity '+' watermark, drop instructions,
    and a Browse button. Shown before any cube is loaded."""

    if QtCore is not None:
        browseRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(32)

        self._plus_frame = QtWidgets.QFrame(self)
        self._plus_frame.setObjectName("plusFrame")
        self._plus_frame.setFixedSize(240, 240)
        frame_layout = QtWidgets.QGridLayout(self._plus_frame)
        frame_layout.setContentsMargins(12, 12, 12, 12)

        self._plus = PlusIcon(self._plus_frame)
        frame_layout.addWidget(self._plus, 0, 0)

        layout.addWidget(self._plus_frame, alignment=QtCore.Qt.AlignCenter)

        row = QtWidgets.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignCenter)
        row.setSpacing(6)

        self._text = QtWidgets.QLabel("Drag and drop FITS/HDF5/NumPy file or", self)
        row.addWidget(self._text)

        self._browse_button = QtWidgets.QPushButton("Browse", self)
        self._browse_button.clicked.connect(self.browseRequested.emit)
        row.addWidget(self._browse_button)

        layout.addLayout(row)

    def apply_theme(self, palette):
        self.setStyleSheet(f"background: {palette['BG']};")
        # Low-opacity accent-on-background watermark, done via rgba rather
        # than a QGraphicsOpacityEffect so it composites against the
        # window's own background colour, not whatever is behind the app.
        accent = QtGui.QColor(palette["ACCENT"])
        self._plus.set_color(QtGui.QColor(accent.red(), accent.green(), accent.blue(), 40))
        self._plus_frame.setStyleSheet(
            f"QFrame#plusFrame {{ border-width: 8px; border-style: dashed; "
            f"border-color: rgba({accent.red()}, {accent.green()}, {accent.blue()}, 25); "
            "border-radius: 12px; background: transparent; }"
        )
        self._text.setStyleSheet(
            f"QLabel {{ color: {palette['TEXT']}; background: transparent; font-family: Georgia, 'Times New Roman'; font-size: 13px; }}"
        )
        self._browse_button.setStyleSheet(f"""
            QPushButton {{
                background: {palette['ENTRY_BG']};
                color: {palette['ACCENT']};
                border: 1px solid {palette['ACCENT']};
                border-radius: 3px;
                padding: 4px 14px;
                font-family: Georgia, 'Times New Roman';
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: {palette['SLIDER_BORDER']};
            }}
        """)


class CubeViewerApp(QtWidgets.QMainWindow if QtWidgets is not None else object):
    """Blank-on-launch window: a drag-and-drop landing page until a cube is
    loaded, then a square interactive PyVista viewport with a right-hand
    controls column (info, clim sliders, opacity-scale pills, theme
    toggle) — never drawn inside the render window itself."""

    def __init__(self):
        super().__init__()
        self._is_dark = True
        self.setWindowTitle("AstroVOX")
        # Square viewport (side == window height) + fixed-width controls
        # column, with no leftover slack — a wider default just left a
        # blank strip between the cube and the column. Fixed (not just
        # defaulted) so the square-viewport/column layout can't be thrown
        # off by a manual resize. The controls column's tallest state is
        # a numpy cube (ManualInfoForm's extra rows on top of everything
        # else) — kept at 844 (rather than growing the window further)
        # by shrinking the Visual Aesthetics pills (see ToggleGrid) just
        # enough for that tallest state to fit without Qt silently
        # compressing row spacing to squeeze it in (seen as uneven gaps
        # between pills, only when a numpy cube was loaded).
        viewport_side = 844
        controls_col_width = 400
        self.setFixedSize(viewport_side + controls_col_width, viewport_side)
        self.setAcceptDrops(True)

        self._stack = QtWidgets.QStackedWidget(self)
        self.setCentralWidget(self._stack)

        # ── Page 0: drop zone (no controls column visible here) ─────────
        self.drop_zone = DropZone(self._stack)
        self.drop_zone.browseRequested.connect(self.browse_for_cube)
        self._stack.addWidget(self.drop_zone)

        # ── Page 1: viewport + controls column ───────────────────────────
        self._loaded_page = QtWidgets.QWidget(self._stack)
        root_layout = QtWidgets.QHBoxLayout(self._loaded_page)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.viewport = SquareViewportContainer(self._loaded_page)
        root_layout.addWidget(self.viewport, 1)

        # Recording indicator: blinking red dot + "REC" + elapsed mm:ss,
        # bottom-right of the viewport, shown only while actually
        # recording (see _start_rec_indicator/_stop_rec_indicator). A
        # plain Qt widget drawn on top of (not inside) the VTK render
        # window, like the capture flash — it doesn't leak into
        # write_frame()'s captured video pixels for the same reason.
        # Colours are theme-dependent (see _update_rec_indicator_theme).
        self._rec_indicator = QtWidgets.QWidget(self.viewport)
        self._rec_indicator.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._rec_indicator.setStyleSheet("background: transparent;")
        self._rec_indicator.hide()
        rec_layout = QtWidgets.QVBoxLayout(self._rec_indicator)
        rec_layout.setContentsMargins(0, 0, 0, 0)
        rec_layout.setSpacing(2)

        rec_row = QtWidgets.QHBoxLayout()
        rec_row.setContentsMargins(0, 0, 0, 0)
        rec_row.setSpacing(4)
        self._rec_dot = QtWidgets.QLabel(self._rec_indicator)
        self._rec_dot.setFixedSize(12, 12)
        rec_row.addWidget(self._rec_dot)
        self._rec_label = QtWidgets.QLabel("REC", self._rec_indicator)
        rec_row.addWidget(self._rec_label)
        rec_layout.addLayout(rec_row)

        self._rec_time_label = QtWidgets.QLabel("00:00", self._rec_indicator)
        rec_layout.addWidget(self._rec_time_label)

        self._rec_dot_on = True
        self._rec_blink_timer = QtCore.QTimer(self)
        self._rec_blink_timer.setInterval(400)
        self._rec_blink_timer.timeout.connect(self._toggle_rec_dot)

        self._rec_start_time = None
        self._rec_elapsed_timer = QtCore.QTimer(self)
        self._rec_elapsed_timer.setInterval(1000)
        self._rec_elapsed_timer.timeout.connect(self._update_rec_elapsed)

        self._controls_col = QtWidgets.QWidget(self._loaded_page)
        self._controls_col.setFixedWidth(controls_col_width)
        controls_layout = QtWidgets.QVBoxLayout(self._controls_col)
        controls_layout.setContentsMargins(14, 10, 14, 10)
        controls_layout.setSpacing(8)

        self.manual_info_form = ManualInfoForm(self._controls_col)
        self.manual_info_form.fovChanged.connect(self._on_manual_fov_changed)
        self.manual_info_form.specResChanged.connect(self._on_manual_specres_changed)
        self.manual_info_form.cubeTypeChanged.connect(self._on_manual_cube_type_changed)
        self.manual_info_form.quantityUnitsChanged.connect(self._on_manual_quantity_units_changed)
        controls_layout.addWidget(self.manual_info_form)
        self._is_numpy_cube = False

        self.vmin_slider = LabeledSlider("V<sub>min</sub>", 0.0, 1.0, 0.0, fmt="{:.2e}", parent=self._controls_col)
        self.vmax_slider = LabeledSlider("V<sub>max</sub>", 0.0, 1.0, 1.0, fmt="{:.2e}", parent=self._controls_col)
        self.vmin_slider.valueChanged.connect(self._on_vmin_changed)
        self.vmax_slider.valueChanged.connect(self._on_vmax_changed)
        controls_layout.addWidget(self.vmin_slider)
        controls_layout.addWidget(self.vmax_slider)

        self.scale_selector = PillSelector(
            ["Linear", "Log", "Power"], selected="Linear", parent=self._controls_col,
            pill_height=26, pill_width=56,
        )
        self.scale_selector.valueChanged.connect(self._on_scale_changed)

        self.gamma_slider = LabeledSlider("γ", 0.1, 6.0, _DEFAULT_GAMMA, fmt="{:.2f}", parent=self._controls_col)
        self.gamma_slider.valueChanged.connect(self._on_gamma_changed)
        self.gamma_slider.set_enabled_dimmed(False)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.setSpacing(8)
        scale_row.addWidget(self.scale_selector)
        scale_row.addWidget(self.gamma_slider, 1)
        controls_layout.addLayout(scale_row)

        self.colormap_selector = ColormapSelector(self._controls_col)
        self.colormap_selector.valueChanged.connect(self._on_cmap_changed)
        controls_layout.addWidget(self.colormap_selector)

        self.axis_snap_label = QtWidgets.QLabel("Snap to axis", self._controls_col)
        controls_layout.addWidget(self.axis_snap_label)

        self.axis_snap_selector = PillSelector(
            ["Free", "X-Y", "Y-Z", "X-Z"], selected="Free", parent=self._controls_col, pill_height=20
        )
        self.axis_snap_selector.valueChanged.connect(self._on_axis_snap_changed)
        controls_layout.addWidget(self.axis_snap_selector)

        controls_layout.addSpacing(6)
        self.aesthetics_card = Card(self._controls_col, faint_border=True)
        aesthetics_content = self.aesthetics_card.layout_for_content()
        self.aesthetics_label = QtWidgets.QLabel("Visual Aesthetics", self.aesthetics_card)
        aesthetics_content.addWidget(self.aesthetics_label)

        self.aesthetics_toggles = ToggleGrid(
            [
                ("Mini Axes", True),
                ("Main Axes Labels", False),
                ("Ticks and Labels", False),
                ("Grid Lines", False),
                ("Colorbar", True),
                ("Scalebar", True),
            ],
            parent=self.aesthetics_card,
        )
        self.aesthetics_toggles.toggled.connect(self._on_aesthetic_toggled)
        aesthetics_content.addWidget(self.aesthetics_toggles)

        self.cube_outline_row = CubeOutlineRow(self.aesthetics_card)
        self.cube_outline_row.toggled.connect(self._on_cube_outline_toggled)
        self.cube_outline_row.thicknessChanged.connect(self._on_cube_outline_thickness_changed)
        self.cube_outline_row.styleChanged.connect(self._on_cube_outline_style_changed)
        aesthetics_content.addWidget(self.cube_outline_row)
        controls_layout.addWidget(self.aesthetics_card)

        controls_layout.addSpacing(6)
        self.animation_card = Card(self._controls_col, faint_border=True)
        animation_content = self.animation_card.layout_for_content()
        self.animation_label = QtWidgets.QLabel("Animation", self.animation_card)
        animation_content.addWidget(self.animation_label)

        self.azimuth_row = PlaybackRow("Azimuth / Horizontal Velocity:", parent=self.animation_card)
        self.azimuth_row.toggled.connect(lambda playing: self._on_animation_toggled("azimuth", playing))
        animation_content.addWidget(self.azimuth_row)

        self.elevation_row = PlaybackRow("Elevation / Vertical Velocity:", parent=self.animation_card)
        self.elevation_row.toggled.connect(lambda playing: self._on_animation_toggled("elevation", playing))
        animation_content.addWidget(self.elevation_row)
        controls_layout.addWidget(self.animation_card)

        self._animation_timer = QtCore.QTimer(self)
        self._animation_timer.setInterval(33)  # ~30 fps
        self._animation_timer.timeout.connect(self._on_animation_tick)
        self._animation_last_ms = None

        controls_layout.addSpacing(6)
        self.export_card = Card(self._controls_col, faint_border=True)
        export_content = self.export_card.layout_for_content()
        self.export_label = QtWidgets.QLabel("Export", self.export_card)
        export_content.addWidget(self.export_label)

        self.record_control = RecordControl(self.export_card)
        self.record_control.recordClicked.connect(self._on_record_clicked)
        self.record_control.countdownFinished.connect(self._on_record_countdown_finished)
        self.record_control.stopClicked.connect(self._on_record_stop_clicked)
        self.record_control.saveClicked.connect(self._on_record_save_clicked)
        self.record_control.resetClicked.connect(self._on_record_reset_clicked)
        export_content.addWidget(self.record_control)

        self._record_frame_timer = QtCore.QTimer(self)
        self._record_frame_timer.setInterval(50)  # 20 fps
        self._record_frame_timer.timeout.connect(self._on_record_frame_tick)
        self._record_tmp_path = None
        self._record_frame_count = 0
        self._record_fail_streak = 0

        self.static_frame_control = StaticFrameControl(self.export_card)
        self.static_frame_control.captureClicked.connect(self._on_static_capture_clicked)
        self.static_frame_control.saveClicked.connect(self._on_static_save_clicked)
        self.static_frame_control.resetClicked.connect(self._on_static_reset_clicked)
        export_content.addWidget(self.static_frame_control)
        controls_layout.addWidget(self.export_card)
        self._captured_frame = None

        # Recording only makes sense while the cube is actually moving,
        # and a "static" frame only makes sense while it's actually
        # still — so each row's own availability is gated by whether
        # either animation row is currently playing, in addition to its
        # own internal click-flow state.
        self.azimuth_row.toggled.connect(self._on_animation_play_state_changed)
        self.elevation_row.toggled.connect(self._on_animation_play_state_changed)
        self._on_animation_play_state_changed(False)

        controls_layout.addStretch(1)

        theme_row = QtWidgets.QHBoxLayout()
        self.reset_button = ResetButton(self._controls_col)
        self.reset_button.clicked.connect(self._on_reset_clicked)
        theme_row.addWidget(self.reset_button)
        theme_row.addStretch(1)
        self.docs_button = LinkButton(
            "API Docs", "https://arnablahiry.github.io/software/AstroVOX/", self._controls_col
        )
        theme_row.addWidget(self.docs_button)
        theme_row.addStretch(1)
        self.github_button = LinkButton(
            "GitHub", "https://github.com/arnablahiry/AstroVOX", self._controls_col
        )
        theme_row.addWidget(self.github_button)
        theme_row.addStretch(1)
        self.theme_button = ThemeButton(self._controls_col)
        self.theme_button.clicked.connect(self.toggle_theme)
        theme_row.addWidget(self.theme_button)
        controls_layout.addLayout(theme_row)

        root_layout.addWidget(self._controls_col)

        self._stack.addWidget(self._loaded_page)
        self._stack.setCurrentWidget(self.drop_zone)

        self.plotter = None
        self.viewer = None

        self._apply_theme()

    # ---------------------------
    # Drag and drop
    # ---------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
            Path(url.toLocalFile()).suffix.lower() in _CUBE_EXTENSIONS for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in _CUBE_EXTENSIONS:
                self.render_cube(path)
                break

    # ---------------------------
    # Cube loading / rendering
    # ---------------------------
    def browse_for_cube(self):
        cube_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select FITS/HDF5/NumPy Cube", "", CUBE_FILE_FILTER,
        )
        if not cube_path:
            return
        # Deferred to the next event-loop tick: render_cube() can open a
        # second native modal dialog (the multi-cube picker, for a
        # .npy/.npz holding more than one cube) — doing that immediately
        # after the native macOS file panel closes, before its own
        # teardown has had a chance to finish, crashes the process.
        QtCore.QTimer.singleShot(0, lambda: self.render_cube(Path(cube_path)))

    def render_cube(self, cube_path: Path):
        self._is_numpy_cube = cube_path.suffix.lower() in _NUMPY_EXTENSIONS

        # A .npy/.npz file can hold more than one volumetric cube (a
        # multi-array .npz, or a single 4D array stacking cubes along
        # axis 0) — ask which one before touching anything, so cancelling
        # leaves whatever's currently loaded untouched.
        cube_index = 0
        if self._is_numpy_cube:
            try:
                count = numpy_cube_count(cube_path)
            except Exception:
                count = 1
            if count > 1:
                chosen, ok = QtWidgets.QInputDialog.getInt(
                    self,
                    "Choose a Cube",
                    f"There are {count} volumetric cubes in this file, choose one:",
                    1, 1, count, 1,
                )
                if not ok:
                    return
                cube_index = chosen - 1

        cube, info, extra = load_cube_with_metadata(cube_path, cube_index=cube_index)

        # Loading a new cube tears down the current plotter/render window —
        # any in-progress recording would be left writing to a now-dead
        # writer, so cut it loose (discarding the unsaved clip) rather than
        # risk it erroring on the next frame tick.
        self._record_frame_timer.stop()
        self._stop_rec_indicator()
        if self._record_tmp_path and Path(self._record_tmp_path).exists():
            try:
                os.remove(self._record_tmp_path)
            except OSError:
                pass
        self._record_tmp_path = None
        self.record_control.reset_idle()
        self._captured_frame = None
        self.static_frame_control.reset()

        if self.plotter is not None:
            self.viewport.clear_child()
            self.plotter.close()
            self.plotter.deleteLater()
            self.plotter = None
            self.viewer = None

        self.plotter = TrackpadInteractor(self.viewport)
        self.viewport.set_child(self.plotter)

        self.viewer = KinematicVolumeViewer(
            cube, plotter=self.plotter, embed_controls=False,
            cmap=self.colormap_selector.current_value(),
            axis_labels=extra["axis_labels"],
            spatial_scale=extra["spatial_scale"],
            colorbar_title=extra.get("colorbar_title", "Intensity"),
            axis_ranges=extra.get("axis_ranges"),
            axis_label_formats=extra.get("axis_label_formats"),
            axis_tick_formatters=extra.get("axis_tick_formatters"),
            axis_tick_units=extra.get("axis_tick_units"),
        )
        self.viewer.set_theme(self._is_dark)
        is_log = self.scale_selector.current_value() == "Log"
        self.vmin_slider.set_log_scale(is_log)
        self.vmax_slider.set_log_scale(is_log)
        self.viewer.set_value_scale("log" if is_log else "linear")
        self.viewer.set_opacity(self._current_opacity_array())
        self.plotter.reset_camera()
        # reset_camera() fits the cube snugly edge-to-edge; back off a
        # little so it starts with some breathing room instead of
        # touching the viewport border.
        self.plotter.renderer.GetActiveCamera().Zoom(0.5)
        self.plotter.set_pivot(self.viewer.grid.center)
        self.axis_snap_selector.set_selected("Free")
        self.plotter.renderer.GetActiveCamera().AddObserver(
            "ModifiedEvent", lambda obj, evt: self._on_camera_modified()
        )

        # A freshly-created viewer starts from its own class defaults —
        # bring it in line with whatever the (persistent, not recreated
        # per cube) Visual aesthetics toggles are currently set to.
        self.viewer.set_mini_axes_visible(self.aesthetics_toggles.is_checked("Mini Axes"))
        self.viewer.set_main_axes_labels_visible(self.aesthetics_toggles.is_checked("Main Axes Labels"))
        self.viewer.set_axis_ticks_visible(self.aesthetics_toggles.is_checked("Ticks and Labels"))
        self.viewer.set_grid_lines_visible(self.aesthetics_toggles.is_checked("Grid Lines"))
        self.viewer.set_colorbar_visible(self.aesthetics_toggles.is_checked("Colorbar"))
        if self._is_numpy_cube:
            # A bare numpy array has no spatial scale at all until the
            # user fills in Field of view — no scale bar to show, and no
            # point letting the pill be clicked in the meantime.
            self.aesthetics_toggles.set_checked_silent("Scalebar", False)
            self.aesthetics_toggles.set_pill_enabled("Scalebar", False)
            self.viewer.set_scalebar_visible(False)
        else:
            self.aesthetics_toggles.set_pill_enabled("Scalebar", True)
            self.viewer.set_scalebar_visible(self.aesthetics_toggles.is_checked("Scalebar"))
        self.viewer.set_cube_outline_visible(self.cube_outline_row.is_checked())
        self.viewer.set_cube_outline_thickness(self.cube_outline_row.thickness())
        self.viewer.set_cube_outline_style(self.cube_outline_row.line_style())
        # Without this, the newly-created render widget can show a stale
        # (blank/white) first frame on macOS until the next repaint tick.
        self.plotter.render()

        d_min, d_max = self.viewer.d_min, self.viewer.d_max
        vmin, vmax = self.viewer.current_clim
        self.vmin_slider.set_range(d_min, d_max, vmin)
        self.vmax_slider.set_range(d_min, d_max, vmax)

        theme_palette = _THEMES["dark"] if self._is_dark else _THEMES["light"]
        # Same editable textbox form for every cube type — pre-filled
        # from whatever the FITS/HDF5 header actually provided (see
        # ManualInfoForm.prefill), left blank/"$$" for a numpy array
        # that carries no metadata at all.
        self.manual_info_form.reset(default_name=cube_path.stem)
        self.manual_info_form.prefill(info, extra)
        self.manual_info_form.apply_theme(theme_palette)

        self._stack.setCurrentWidget(self._loaded_page)

        # The scale bar and colorbar border/ticks are positioned from the
        # render window's *current* pixel size — but the interactor
        # widget was only just created this call, and Qt doesn't actually
        # resize its render window to the widget's final geometry until
        # the event loop processes the pending resize/paint events. Doing
        # this synchronously here would still see a stale placeholder
        # size; deferring it to the next event-loop tick (after Qt has
        # settled) is what actually gets the final, correct size.
        QtCore.QTimer.singleShot(0, self._finalize_cube_view)

    def _finalize_cube_view(self):
        if self.viewer is None:
            return
        self.viewer._update_scale_bar()
        self.viewer._rebuild_colorbar()
        self.plotter.render()
        # The very first _apply_theme() call happens in __init__, before
        # the window has ever been shown — a stylesheet set on a widget
        # that isn't on screen yet gets cached with wrong-looking colors
        # (pill backgrounds render too dark) until something re-applies it
        # post-show, which is why a manual theme toggle "fixes" it: that
        # setStyleSheet call happens on an already-visible widget. Now that
        # the loaded page is actually on screen, re-run it once so launch
        # matches a post-toggle look exactly instead of merely re-polishing
        # the checked-state highlight (which isn't the whole story here).
        self._apply_theme()
        # cube_outline_row.set_toggle_width() and manual_info_form's own
        # pill-height sync (inside _apply_theme) each read another
        # widget's actual just-restyled size — but a size measured in
        # the very same call that changed its stylesheet can still be
        # stale (Qt only reflows on the next event-loop tick), so this
        # takes a couple more passes to fully converge.
        QtCore.QTimer.singleShot(0, self._apply_theme)
        QtCore.QTimer.singleShot(0, self._apply_theme)

    def _on_vmin_changed(self, value):
        if self.viewer is None:
            return
        vmax = self.viewer.current_clim[1]
        if value < vmax:
            self.viewer.set_clim(value, vmax)

    def _on_vmax_changed(self, value):
        if self.viewer is None:
            return
        vmin = self.viewer.current_clim[0]
        if value > vmin:
            self.viewer.set_clim(vmin, value)

    def _current_opacity_array(self):
        scale = self.scale_selector.current_value() or "Linear"
        if scale == "Linear":
            return _linear_opacity()
        if scale == "Log":
            return _log_opacity()
        return _power_opacity(self.gamma_slider.value())

    def _on_scale_changed(self, name):
        self.gamma_slider.set_enabled_dimmed(name == "Power")
        is_log = name == "Log"
        self.vmin_slider.set_log_scale(is_log)
        self.vmax_slider.set_log_scale(is_log)
        if self.viewer is None:
            return
        self.viewer.set_value_scale("log" if is_log else "linear")
        self.viewer.set_opacity(self._current_opacity_array())

    def _on_gamma_changed(self, _value):
        if self.viewer is None or self.scale_selector.current_value() != "Power":
            return
        self.viewer.set_opacity(self._current_opacity_array())

    def _on_cmap_changed(self, name):
        if self.viewer is None:
            return
        self.viewer.set_cmap(name)

    def _on_axis_snap_changed(self, name):
        if self.plotter is None or name == "Free":
            return
        self.plotter.snap_to_axis_plane(name)

    def _on_camera_modified(self):
        # Any camera change NOT caused by clicking a snap pill itself
        # (manual rotate/pan/zoom, whether via mouse or trackpad) means
        # the view is no longer exactly on-axis — fall back to "Free"
        # without re-triggering a snap.
        if getattr(self.plotter, "_suppress_free_revert", False):
            return
        if self.axis_snap_selector.current_value() != "Free":
            self.axis_snap_selector.set_selected("Free")

    def _on_aesthetic_toggled(self, name, state):
        if self.viewer is None:
            return
        if name == "Mini Axes":
            self.viewer.set_mini_axes_visible(state)
        elif name == "Main Axes Labels":
            self.viewer.set_main_axes_labels_visible(state)
        elif name == "Ticks and Labels":
            self.viewer.set_axis_ticks_visible(state)
        elif name == "Grid Lines":
            self.viewer.set_grid_lines_visible(state)
        elif name == "Colorbar":
            self.viewer.set_colorbar_visible(state)
        elif name == "Scalebar":
            self.viewer.set_scalebar_visible(state)

    def _on_manual_fov_changed(self, xyz, unit):
        """Field of view (now three independent X/Y/Z side lengths, not
        one value assumed to apply to every axis) only feeds the scale
        bar once X and a unit are both present — the Scalebar pill is
        gated on that same condition (see render_cube's numpy-cube
        branch for the initial disabled state). Also feeds each axis's
        own tick labels + title once given (see
        KinematicVolumeViewer.set_manual_axis_scale). Z is only
        meaningful for a PPP cube (see ManualInfoForm._set_fov_z_enabled)
        — for anything else it's the velocity axis, governed by
        Spectral Resolution instead."""
        if self.viewer is None:
            return
        x, y, z = xyz
        is_ppp = self.manual_info_form.cube_type() == "PPP"
        dims = self.viewer.grid.dimensions  # (nx, ny, nz)

        if x is not None and unit:
            # "value" is the *total* Field of view across the cube, not
            # a per-voxel scale — set_spatial_scale/set_manual_axis_scale
            # both need the latter, so divide by X's own voxel count.
            value_per_voxel_x = x / dims[0] if dims[0] else x
            self.viewer.set_spatial_scale(value_per_voxel_x, unit)
            self.aesthetics_toggles.set_pill_enabled("Scalebar", True)
            self.aesthetics_toggles.set_checked_silent("Scalebar", True)
            self.viewer.set_scalebar_visible(True)
        else:
            self.aesthetics_toggles.set_checked_silent("Scalebar", False)
            self.aesthetics_toggles.set_pill_enabled("Scalebar", False)
            self.viewer.set_scalebar_visible(False)

        # Only a numpy cube's axes are generic voxel-index ticks waiting
        # to be given units — a FITS/HDF5 cube already has its own
        # correct RA/Dec (or kpc) tick system, which this would
        # incorrectly stomp over if applied there too (the form is
        # pre-filled and editable for those cubes as well, purely for
        # display/override of the *other* fields).
        if not self._is_numpy_cube:
            return
        for axis_idx, val in ((0, x), (1, y), (2, z if is_ppp else None)):
            if val is not None and unit:
                value_per_voxel = val / dims[axis_idx] if dims[axis_idx] else val
                self.viewer.set_manual_axis_scale(axis_idx, value_per_voxel, unit)
            elif axis_idx != 2 or is_ppp:
                self.viewer.clear_manual_axis_scale(axis_idx)

    def _on_manual_specres_changed(self, value=None, unit=None):
        """Spectral Resolution feeds the depth (Z) axis's tick labels +
        title once both a numeric value and a unit are present — it's
        already a per-channel quantity, so unlike Field of view it's
        used directly rather than divided by a voxel count. Also
        re-invoked (with no args, re-reading the form directly) when
        Type of cube changes, since switching to/from PPP flips whether
        this field is even meaningful."""
        if self.viewer is None:
            return
        if value is None and unit is None:
            form = self.manual_info_form
            text = form._specres_value_edit.text().strip()
            unit = form._specres_unit_edit.text().strip()
            try:
                value = float(text) if text else None
            except ValueError:
                value = None
        if not self._is_numpy_cube:
            # A FITS/HDF5 cube's Z axis already has its own correct
            # (velocity, or raw-voxel for PPP) tick system set up at
            # load time — this form is pre-filled/editable for such
            # cubes too, but only a numpy cube's axes are blank ticks
            # actually waiting to be given units here.
            return
        if value is not None and unit and self.manual_info_form.cube_type() != "PPP":
            self.viewer.set_manual_axis_scale(2, value, unit, centered=True)
        else:
            self.viewer.clear_manual_axis_scale(2)

    def _on_manual_cube_type_changed(self, _value):
        self._on_manual_specres_changed()

    def _on_manual_quantity_units_changed(self, text):
        if self.viewer is not None:
            self.viewer.set_colorbar_title(text)

    def _on_cube_outline_toggled(self, state):
        if self.viewer is not None:
            self.viewer.set_cube_outline_visible(state)

    def _on_cube_outline_thickness_changed(self, thickness):
        if self.viewer is not None:
            self.viewer.set_cube_outline_thickness(thickness)

    def _on_cube_outline_style_changed(self, style):
        if self.viewer is not None:
            self.viewer.set_cube_outline_style(style)

    def _on_animation_toggled(self, axis, playing):
        if not playing and not (self.azimuth_row.is_playing() or self.elevation_row.is_playing()):
            self._animation_timer.stop()
            self._animation_last_ms = None
            return
        self._animation_last_ms = None
        if not self._animation_timer.isActive():
            self._animation_timer.start()

    def _on_animation_play_state_changed(self, _playing=None):
        # Recording a "video" only makes sense while the cube is actually
        # moving, and a "static frame" only makes sense while it's
        # actually still — so the two rows are mutually gated on whether
        # either animation row is currently playing.
        any_playing = self.azimuth_row.is_playing() or self.elevation_row.is_playing()
        self.record_control.set_animation_gate(any_playing)
        self.static_frame_control.set_animation_gate(not any_playing)

    def _on_static_capture_clicked(self):
        if self.plotter is None:
            return
        # Defer the actual VTK render/screenshot to the next event-loop tick so
        # it doesn't run re-entrantly inside the button's mousePressEvent while
        # Qt is still mid-repaint from the click's own style changes (this was
        # corrupting nearby widget text and emitting QPainter warnings).
        QtCore.QTimer.singleShot(0, self._do_capture_static_frame)

    def _do_capture_static_frame(self):
        if self.plotter is None:
            return
        try:
            self._captured_frame = self.plotter.screenshot(return_img=True)
        except Exception:
            print("Failed to capture the current frame.")
            traceback.print_exc()
            self._captured_frame = None
            return
        self._flash_viewport()

    def _flash_viewport(self):
        """Camera-shutter flash over the viewport on capture — a plain
        white overlay widget, on screen just long enough to read as a
        flash, then torn down. Drawn as a Qt widget on top of (not
        inside) the VTK render window, so it never leaks into the
        screenshot pixel data taken just before this runs."""
        flash = QtWidgets.QWidget(self.viewport)
        flash.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        flash.setStyleSheet("background: rgba(255, 255, 255, 0.35);")
        flash.setGeometry(self.viewport.rect())
        flash.show()
        flash.raise_()
        QtCore.QTimer.singleShot(150, flash.deleteLater)

    def _default_export_name(self, kind: str) -> str:
        """e.g. "IC5179_recording_20260823_143012" — falls back to
        "cube" when the Name field is blank (e.g. an unfilled numpy
        cube)."""
        name = self.manual_info_form.name() or "cube"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{kind}_{stamp}"

    def _on_static_reset_clicked(self):
        self._captured_frame = None

    def _on_static_save_clicked(self, fmt):
        if self._captured_frame is None:
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Save", "No captured frame is available — try capturing again."
            )
            return
        filters = {
            "png": "PNG Image (*.png)",
            "jpg": "JPEG Image (*.jpg *.jpeg)",
            "pdf": "PDF Document (*.pdf)",
            "tiff": "TIFF Image (*.tiff *.tif)",
        }
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Frame",
            str(Path.home() / f"{self._default_export_name('capture')}.{fmt}"),
            filters.get(fmt, "All Files (*)"),
        )
        if not dest:
            return
        if not dest.lower().endswith(f".{fmt}"):
            dest += f".{fmt}"
        try:
            from PIL import Image

            image = Image.fromarray(self._captured_frame)
            if fmt == "pdf":
                image.convert("RGB").save(dest, "PDF")
            else:
                image.save(dest)
        except Exception:
            print(f"Failed to save frame to {dest!r}.")
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Save Failed", f"Could not save the frame to:\n{dest}")

    def _on_animation_tick(self):
        if self.viewer is None or self.plotter is None:
            return
        now = time.monotonic()
        if self._animation_last_ms is None:
            self._animation_last_ms = now
            return
        dt = now - self._animation_last_ms
        self._animation_last_ms = now

        camera = self.plotter.renderer.GetActiveCamera()
        rotated = False
        if self.azimuth_row.is_playing():
            camera.Azimuth(self.azimuth_row.speed() * dt)
            rotated = True
        if self.elevation_row.is_playing():
            camera.Elevation(self.elevation_row.speed() * dt)
            camera.OrthogonalizeViewUp()
            rotated = True
        if rotated:
            self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()

    # ---------------------------
    # Recording
    # ---------------------------

    def _rec_colors(self):
        """(red, time-text colour) for the current theme — a lighter red
        reads clearly against the dark viewport, a darker red against
        the light one; the elapsed-time text follows the same logic."""
        if self._is_dark:
            return "#ff6b6b", "white"
        return "#a30000", "black"

    def _update_rec_indicator_theme(self):
        red, time_color = self._rec_colors()
        self._rec_label.setStyleSheet(
            f"color: {red}; font-family: 'Courier New', monospace; font-size: 15px; "
            "font-weight: bold; background: transparent;"
        )
        self._rec_time_label.setStyleSheet(
            f"color: {time_color}; font-family: 'Courier New', monospace; font-size: 15px; background: transparent;"
        )
        self._rec_dot.setStyleSheet(
            f"background: {red if self._rec_dot_on else 'transparent'}; border-radius: 6px;"
        )

    def _toggle_rec_dot(self):
        self._rec_dot_on = not self._rec_dot_on
        red, _ = self._rec_colors()
        color = red if self._rec_dot_on else "transparent"
        self._rec_dot.setStyleSheet(f"background: {color}; border-radius: 6px;")

    def _update_rec_elapsed(self):
        if self._rec_start_time is None:
            return
        elapsed = int(time.monotonic() - self._rec_start_time)
        mins, secs = divmod(elapsed, 60)
        self._rec_time_label.setText(f"{mins:02d}:{secs:02d}")

    def _start_rec_indicator(self):
        self._rec_start_time = time.monotonic()
        self._rec_dot_on = True
        self._update_rec_indicator_theme()
        self._rec_time_label.setText("00:00")
        self._rec_indicator.adjustSize()
        margin = 12
        self._rec_indicator.move(
            self.viewport.width() - self._rec_indicator.width() - margin,
            self.viewport.height() - self._rec_indicator.height() - margin,
        )
        self._rec_indicator.show()
        self._rec_indicator.raise_()
        self._rec_blink_timer.start()
        self._rec_elapsed_timer.start()
        self._set_theme_button_enabled(False)

    def _stop_rec_indicator(self):
        self._rec_blink_timer.stop()
        self._rec_elapsed_timer.stop()
        self._rec_start_time = None
        self._rec_indicator.hide()
        self._set_theme_button_enabled(True)

    def _set_theme_button_enabled(self, enabled: bool):
        # Switching theme mid-recording would restyle the in-progress
        # indicator's colours out from under it — simplest to just block
        # the toggle for the duration, same as animation/static-capture
        # gating each other via set_animation_gate.
        self.theme_button.setEnabled(enabled)
        effect = self.theme_button.graphicsEffect()
        if not isinstance(effect, QtWidgets.QGraphicsOpacityEffect):
            effect = QtWidgets.QGraphicsOpacityEffect(self.theme_button)
            self.theme_button.setGraphicsEffect(effect)
        effect.setOpacity(1.0 if enabled else 0.35)

    def _on_record_clicked(self):
        self.record_control.start_countdown(3)

    def _on_record_countdown_finished(self):
        if self.plotter is None:
            return
        fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        self._record_frame_count = 0
        self._record_fail_streak = 0
        try:
            # format="FFMPEG" is forced explicitly rather than left to
            # imageio's extension-based auto-detection — on at least one
            # tested environment (a separate conda env without a properly
            # registered ffmpeg plugin), get_writer() silently fell back to
            # the *tifffile* plugin for a ".mp4" path instead of raising,
            # and every frame then failed with a cryptic
            # "TiffWriter.write() got an unexpected keyword argument 'fps'"
            # deep inside imageio. Forcing the format means a missing/broken
            # ffmpeg plugin now fails loudly and immediately, right here,
            # with a clear message — instead of quietly writing frames to
            # a plugin that can't actually produce a video.
            #
            # -movflags +faststart moves the mp4 "moov" index to the front
            # of the file once ffmpeg finalizes it — without this, ffmpeg's
            # default placement (index at the *end*) plays fine in
            # ffmpeg/VLC but is a common cause of "recorded video won't
            # open" in QuickTime/Preview/Finder Quick Look on macOS.
            self.plotter.open_movie(
                tmp_path,
                framerate=self.record_control.fps(),
                format="FFMPEG",
                output_params=["-movflags", "+faststart"],
                # Viewport frames are whatever the render window's own
                # pixel size happens to be, which usually isn't a
                # multiple of 16 — ffmpeg's default macro_block_size
                # would silently pad/resize every frame to the nearest
                # multiple and print a warning each time. macro_block_size=1
                # disables that padding, matching the frame size exactly.
                macro_block_size=1,
            )
        except Exception:
            print("Failed to start video recording (couldn't open an FFMPEG writer):")
            traceback.print_exc()
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            QtWidgets.QMessageBox.warning(
                self,
                "Recording Unavailable",
                "Could not start video recording — the ffmpeg video backend isn't available "
                "in this Python environment.\n\nTry: pip install imageio imageio-ffmpeg",
            )
            self.record_control.reset_idle()
            return
        self._record_tmp_path = tmp_path
        self._record_frame_timer.setInterval(round(1000 / self.record_control.fps()))
        self._record_frame_timer.start()
        self.record_control.enter_recording()
        self._start_rec_indicator()

    def _on_record_frame_tick(self):
        if self.plotter is None:
            self._record_frame_timer.stop()
            self._stop_rec_indicator()
            return
        try:
            self.plotter.write_frame()
            self._record_frame_count += 1
            self._record_fail_streak = 0
        except Exception:
            # Tolerate a handful of failures in a row rather than aborting
            # on the very first one — e.g. the ffmpeg subprocess spawned
            # by open_movie() may not have finished starting up yet when
            # the first tick or two fires, which previously looked like
            # "recording instantly fails" even though it was just a
            # startup race, not a real, persistent problem.
            self._record_fail_streak += 1
            print(f"Recording frame capture failed (attempt {self._record_fail_streak}):")
            traceback.print_exc()
            if self._record_fail_streak >= 10:
                print("Too many consecutive frame-capture failures; stopping recording.")
                self._record_frame_timer.stop()
                self.record_control._on_stop_clicked()

    def _on_record_stop_clicked(self):
        self._record_frame_timer.stop()
        self._stop_rec_indicator()
        mwriter = getattr(self.plotter, "mwriter", None) if self.plotter is not None else None
        if mwriter is not None:
            try:
                mwriter.close()
            except Exception:
                print("Failed to finalize the recorded video file.")
                traceback.print_exc()
        if self._record_frame_count == 0 and self._record_tmp_path:
            # Nothing was ever captured (e.g. stopped during the same tick
            # it started) — there's no valid video to save.
            try:
                os.remove(self._record_tmp_path)
            except OSError:
                pass
            self._record_tmp_path = None

    def _on_record_reset_clicked(self):
        # Clicking "Reset" (the stopped-state accent box) discards the
        # unsaved clip and returns to idle — Save .mp4 is the only path
        # that actually keeps a recording.
        if self._record_tmp_path and Path(self._record_tmp_path).exists():
            try:
                os.remove(self._record_tmp_path)
            except OSError:
                pass
        self._record_tmp_path = None

    def _on_record_save_clicked(self):
        if not self._record_tmp_path or not Path(self._record_tmp_path).exists():
            QtWidgets.QMessageBox.warning(
                self, "Nothing to Save", "No recorded video is available — try recording again."
            )
            self.record_control.reset_idle()
            return
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Recording",
            str(Path.home() / f"{self._default_export_name('recording')}.mp4"),
            "MP4 Video (*.mp4)",
        )
        if not dest:
            return
        if not dest.lower().endswith(".mp4"):
            dest += ".mp4"
        try:
            shutil.move(self._record_tmp_path, dest)
        except Exception:
            print(f"Failed to save recording to {dest!r}.")
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Save Failed", f"Could not save the recording to:\n{dest}")
            return
        finally:
            self._record_tmp_path = None
            self.record_control.reset_idle()

    # ---------------------------
    # Theme
    # ---------------------------
    def _on_reset_clicked(self):
        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset",
            "Are you sure? This will go back to the upload page.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._animation_timer.stop()
        self._animation_last_ms = None
        self.azimuth_row.stop()
        self.elevation_row.stop()

        self._record_frame_timer.stop()
        self._stop_rec_indicator()
        if self._record_tmp_path and Path(self._record_tmp_path).exists():
            try:
                os.remove(self._record_tmp_path)
            except OSError:
                pass
        self._record_tmp_path = None
        self.record_control.reset_idle()
        self._captured_frame = None
        self.static_frame_control.reset()

        if self.plotter is not None:
            self.viewport.clear_child()
            self.plotter.close()
            self.plotter.deleteLater()
            self.plotter = None
            self.viewer = None

        # Always land back on the dark theme, regardless of what the
        # user had selected before resetting — a fresh, known starting
        # point rather than carrying light mode (which the landing
        # page/controls were never as thoroughly themed for) forward.
        self._is_dark = True
        self._apply_theme()

        self._stack.setCurrentWidget(self.drop_zone)

    def toggle_theme(self):
        self._is_dark = not self._is_dark
        self._apply_theme()
        if self.viewer is not None:
            self.viewer.is_dark_theme = self._is_dark
            self.viewer.cmap = self.colormap_selector.current_value()
        # Qt's stylesheet-driven repaint on the column is queued, not
        # immediate — it only actually reaches the screen once the event
        # loop gets control back. The VTK render() below swaps its OpenGL
        # buffer synchronously and instantly, before that happens, which
        # is exactly why the viewport visibly flips a frame ahead of the
        # rest of the column. Flushing the pending Qt paint here, right
        # before the single VTK render, brings both on screen together.
        QtWidgets.QApplication.processEvents()
        if self.viewer is not None:
            self.viewer.apply_theme()

    def _apply_theme(self):
        palette = _THEMES["dark"] if self._is_dark else _THEMES["light"]
        theme_name = "dark" if self._is_dark else "light"

        self.drop_zone.apply_theme(palette)
        self._loaded_page.setStyleSheet(f"background: {palette['BG']};")
        self._controls_col.setStyleSheet(f"background: {palette['BG']};")
        self.viewport.setStyleSheet(f"background: {palette['BG']};")

        self.manual_info_form.apply_theme(palette)
        self.colormap_selector.set_theme_maps(theme_name)
        self.colormap_selector.apply_theme(palette)
        self.vmin_slider.apply_theme(palette)
        self.vmax_slider.apply_theme(palette)
        self.scale_selector.apply_theme(palette)
        self.gamma_slider.apply_theme(palette)
        heading_css = (
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; font-weight: bold; }}"
        )
        self.axis_snap_label.setStyleSheet(
            f"QLabel {{ color: {palette['CARD_TEXT']}; background: transparent; "
            f"font-family: Georgia, 'Times New Roman'; font-size: 11px; }}"
        )
        self.axis_snap_selector.apply_theme(palette)
        self.aesthetics_card.apply_theme(palette)
        self.aesthetics_label.setStyleSheet(heading_css)
        self.aesthetics_toggles.apply_theme(palette)
        self.cube_outline_row.set_toggle_width(self.aesthetics_toggles.column_width())
        self.cube_outline_row.set_toggle_height(self.aesthetics_toggles.pill_height())
        self.cube_outline_row.apply_theme(palette)
        self.animation_card.apply_theme(palette)
        self.animation_label.setStyleSheet(heading_css)
        self.azimuth_row.apply_theme(palette)
        self.elevation_row.apply_theme(palette)
        self.export_card.apply_theme(palette)
        self.export_label.setStyleSheet(heading_css)
        self.record_control.apply_theme(palette)
        self.static_frame_control.apply_theme(palette)
        self.theme_button.apply_theme(palette, self._is_dark)
        self.reset_button.apply_theme(palette)
        self.docs_button.apply_theme(palette)
        self.github_button.apply_theme(palette)
        self._update_rec_indicator_theme()
        _set_titlebar_theme(self, self._is_dark)

    def closeEvent(self, event):
        self._record_frame_timer.stop()
        if self._record_tmp_path and Path(self._record_tmp_path).exists():
            try:
                os.remove(self._record_tmp_path)
            except OSError:
                pass
        if self.plotter is not None:
            self.plotter.close()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    if QtWidgets is None:
        raise SystemExit(
            f"PyQt5 is not available: {QT_IMPORT_ERROR}. Install it with `pip install PyQt5`."
        )
    if QtInteractor is None:
        raise SystemExit(
            f"pyvistaqt is not available: {PYVISTAQT_IMPORT_ERROR}. Install it with `pip install pyvistaqt`."
        )

    parser = argparse.ArgumentParser(description="AstroVOX — interactive 3D volumetric cube viewer.")
    parser.add_argument(
        "cube_path", type=Path, nargs="?", default=None,
        help="FITS/HDF5/NumPy cube to load on launch (optional — omit to start on the drop-zone landing page)",
    )
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Georgia", 11))

    # Qt's C++ event loop never yields back to the Python interpreter on
    # its own, so the default SIGINT handler (installed by Python) never
    # gets a chance to run — Ctrl+C in the terminal is silently swallowed
    # and the window stays open. Restoring the default handler plus a
    # short-interval timer (which *does* hand control back to Python
    # periodically, letting the pending signal be noticed) fixes it.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sigint_timer = QtCore.QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    icon_path = Path(__file__).resolve().parent.parent.parent / "assets" / "volumetric_ico.png"
    if icon_path.exists():
        icon = QtGui.QIcon(str(icon_path))
        app.setWindowIcon(icon)

    window = CubeViewerApp()
    if icon_path.exists():
        window.setWindowIcon(icon)
    window.show()
    if args.cube_path is not None:
        # Deferred to the next event-loop tick, same as browse_for_cube —
        # loading a cube immediately, before the freshly-shown window has
        # finished settling, is the same crash-prone timing this pattern
        # avoids there.
        QtCore.QTimer.singleShot(0, lambda: window.render_cube(args.cube_path))
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
