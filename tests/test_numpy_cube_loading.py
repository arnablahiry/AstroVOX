"""load_cube_with_metadata()/numpy_cube_count() for .npy/.npz files —
the path a bare numpy array (no FITS/HDF5 header) takes through the
loader, exercised end-to-end against real temp files on disk."""

import numpy as np
import pytest

from astrovox.viewer import load_cube_with_metadata, numpy_cube_count


def test_single_npy_cube_loads_with_generic_axis_labels(tmp_path):
    cube = np.random.rand(4, 5, 6).astype(np.float32)
    path = tmp_path / "cube.npy"
    np.save(path, cube)

    assert numpy_cube_count(path) == 1

    loaded, info, extra = load_cube_with_metadata(path)
    assert loaded.shape == cube.shape
    np.testing.assert_allclose(loaded, cube, rtol=1e-5)
    assert info == {}
    assert extra["axis_labels"] == ("X", "Y", "Z")
    assert extra["spatial_scale"] is None


def test_2d_array_gets_a_singleton_third_axis(tmp_path):
    plane = np.random.rand(8, 9).astype(np.float32)
    path = tmp_path / "plane.npy"
    np.save(path, plane)

    loaded, _, _ = load_cube_with_metadata(path)
    assert loaded.shape == (1, 8, 9)


def test_4d_array_stacks_multiple_cubes_along_axis_0(tmp_path):
    stack = np.random.rand(3, 4, 5, 6).astype(np.float32)
    path = tmp_path / "stack.npy"
    np.save(path, stack)

    assert numpy_cube_count(path) == 3

    for index in range(3):
        loaded, _, _ = load_cube_with_metadata(path, cube_index=index)
        np.testing.assert_allclose(loaded, stack[index], rtol=1e-5)


def test_multi_array_npz_is_indexed_by_key_order(tmp_path):
    cube_a = np.random.rand(2, 3, 4).astype(np.float32)
    cube_b = np.random.rand(2, 3, 4).astype(np.float32) + 10
    path = tmp_path / "cubes.npz"
    np.savez(path, first=cube_a, second=cube_b)

    assert numpy_cube_count(path) == 2

    loaded_first, _, _ = load_cube_with_metadata(path, cube_index=0)
    loaded_second, _, _ = load_cube_with_metadata(path, cube_index=1)
    np.testing.assert_allclose(loaded_first, cube_a, rtol=1e-5)
    np.testing.assert_allclose(loaded_second, cube_b, rtol=1e-5)


def test_single_array_npz_reports_one_cube(tmp_path):
    cube = np.random.rand(2, 3, 4).astype(np.float32)
    path = tmp_path / "single.npz"
    np.savez(path, only=cube)

    assert numpy_cube_count(path) == 1
    loaded, _, _ = load_cube_with_metadata(path)
    np.testing.assert_allclose(loaded, cube, rtol=1e-5)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cube_with_metadata(tmp_path / "nope.npy")
