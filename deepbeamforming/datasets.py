"""PyTorch datasets for toy DAS-IQ deep beamforming."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from scipy.io import loadmat as scipy_loadmat
except ImportError:  # pragma: no cover
    scipy_loadmat = None

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


def _mat_struct_to_dict(obj: Any) -> Any:
    """Recursively convert scipy MATLAB structs into Python containers."""
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return _mat_struct_to_dict(obj[0])
        return [_mat_struct_to_dict(item) for item in obj]

    if isinstance(obj, np.ndarray):
        if obj.dtype.names is not None:
            if obj.size == 1:
                return _mat_struct_to_dict(obj.reshape(-1)[0])
            return [_mat_struct_to_dict(item) for item in obj.reshape(-1)]
        if obj.dtype == object:
            return [_mat_struct_to_dict(item) for item in obj.reshape(-1)]
        return np.array(obj)

    if hasattr(obj, "_fieldnames"):
        return {name: _mat_struct_to_dict(getattr(obj, name)) for name in obj._fieldnames}

    if isinstance(obj, np.void) and obj.dtype.names is not None:
        return {name: _mat_struct_to_dict(obj[name]) for name in obj.dtype.names}

    return obj


def _load_mat_v7(path: Path) -> Dict[str, Any]:
    if scipy_loadmat is None:
        raise ImportError("scipy is required to read non-v7.3 MAT files")

    data = scipy_loadmat(path, squeeze_me=False, struct_as_record=False)
    cleaned = {key: value for key, value in data.items() if not key.startswith("__")}
    if "sample" not in cleaned:
        raise KeyError(f"'sample' variable not found in {path}")
    sample = _mat_struct_to_dict(cleaned["sample"])
    if isinstance(sample, list):
        sample = sample[0]
    return sample


def _read_h5_dataset(node: Any) -> Any:
    if isinstance(node, h5py.Dataset):
        data = node[()]
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return np.array(data)

    if isinstance(node, h5py.Group):
        result: Dict[str, Any] = {}
        for key, value in node.items():
            result[key] = _read_h5_dataset(value)
        if len(result) == 1 and "data" in result:
            return result["data"]
        return result

    return node


def _load_mat_v73(path: Path) -> Dict[str, Any]:
    if h5py is None:
        raise ImportError("h5py is required to read v7.3 MAT files")

    with h5py.File(path, "r") as f:
        if "sample" not in f:
            raise KeyError(f"'sample' variable not found in {path}")
        sample_group = f["sample"]
        if isinstance(sample_group, h5py.Group):
            sample = _read_h5_dataset(sample_group)
        else:
            sample = sample_group[()]
    return sample


def load_sample_mat(path: Path) -> Dict[str, Any]:
    """Load a MATLAB sample struct from either v7 or v7.3 files."""
    try:
        return _load_mat_v7(path)
    except NotImplementedError:
        return _load_mat_v73(path)
    except ValueError as exc:
        # scipy raises ValueError for v7.3 files in some versions.
        message = str(exc).lower()
        if "mat 7.3" in message or "hdf5" in message:
            return _load_mat_v73(path)
        raise


def _to_complex_array(value: Any, field_name: str, path: Path) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        raise ValueError(f"{field_name} is empty in {path}")
    if np.iscomplexobj(array):
        return np.asarray(array, dtype=np.complex128)
    if array.ndim >= 1 and array.shape[-1] == 2 and not np.iscomplexobj(array):
        # Not expected for the current dataset, but keep a defensive fallback.
        return array[..., 0].astype(np.complex128) + 1j * array[..., 1].astype(np.complex128)
    raise ValueError(f"{field_name} is not complex-valued in {path}")


def _unwrap_singleton(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _extract_metadata(sample: Dict[str, Any], path: Path) -> Dict[str, Any]:
    channel_data = _unwrap_singleton(sample.get("channel_data", {}))
    das = _unwrap_singleton(sample.get("das", {}))

    channel_iq_norm = _to_complex_array(channel_data["channel_iq_norm"], "channel_data.channel_iq_norm", path)
    das_iq_norm = _to_complex_array(das["das_iq_norm"], "das.das_iq_norm", path)

    metadata = {
        "file_name": path.name,
        "sx": float(np.asarray(channel_data.get("sx", np.nan)).squeeze()),
        "sy": float(np.asarray(das.get("sy", np.nan)).squeeze()),
        "T": int(channel_iq_norm.shape[0]),
        "C": int(channel_iq_norm.shape[1]),
        "H": int(das_iq_norm.shape[0]),
        "W": int(das_iq_norm.shape[1]),
    }
    return metadata


def _axis_to_1d(value: Any, expected_len: int, field_name: str, path: Path) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).squeeze()
    if axis.ndim != 1 or axis.size != expected_len:
        raise ValueError(
            f"{field_name} must have length {expected_len} for {path}, got shape {axis.shape}"
        )
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{field_name} contains NaN/Inf in {path}")
    return axis


def _normalize_axis(axis: np.ndarray, field_name: str, path: Path) -> np.ndarray:
    axis_min = float(np.min(axis))
    axis_max = float(np.max(axis))
    if axis_max <= axis_min:
        raise ValueError(f"{field_name} has degenerate range in {path}")
    return 2.0 * (axis - axis_min) / (axis_max - axis_min) - 1.0


def _extract_coord_maps(
    das: Dict[str, Any],
    h: int,
    w: int,
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return coordinate maps and physical axes.

    coords is [2, H, W] = [z_local, x_local] for backwards compatibility.
    coords_local_global is [4, H, W] = [z_local, x_local, z_global, x_global].
    Global coordinates use fixed limits in millimeters.
    """
    x_global_min_mm = -20.0
    x_global_max_mm = 20.0
    z_global_min_mm = 0.0
    z_global_max_mm = 60.0

    if "x_axis" in das and "z_axis" in das:
        x_axis_m = _axis_to_1d(das["x_axis"], w, "das.x_axis", path)
        z_axis_m = _axis_to_1d(das["z_axis"], h, "das.z_axis", path)
        x_axis_mm = x_axis_m * 1e3
        z_axis_mm = z_axis_m * 1e3
        x_local = _normalize_axis(x_axis_m, "das.x_axis", path)
        z_local = _normalize_axis(z_axis_m, "das.z_axis", path)
        coord_source = "physical_axes"
    else:
        print(f"Warning: physical x_axis/z_axis not found in {path}; using index fallback for plots and coordinates.")
        x_axis_mm = np.arange(w, dtype=np.float64)
        z_axis_mm = np.arange(h, dtype=np.float64)
        x_local = np.linspace(-1.0, 1.0, w, dtype=np.float64)
        z_local = np.linspace(-1.0, 1.0, h, dtype=np.float64)
        coord_source = "fallback_indices"

    x_global = 2.0 * (x_axis_mm - x_global_min_mm) / (x_global_max_mm - x_global_min_mm) - 1.0
    z_global = 2.0 * (z_axis_mm - z_global_min_mm) / (z_global_max_mm - z_global_min_mm) - 1.0

    z_local_map = np.repeat(z_local[:, None], w, axis=1)
    x_local_map = np.repeat(x_local[None, :], h, axis=0)
    z_global_map = np.repeat(z_global[:, None], w, axis=1)
    x_global_map = np.repeat(x_global[None, :], h, axis=0)

    coords = np.stack([z_local_map, x_local_map], axis=0).astype(np.float32, copy=False)
    coords_local_global = np.stack(
        [z_local_map, x_local_map, z_global_map, x_global_map],
        axis=0,
    ).astype(np.float32, copy=False)

    coord_metadata = {
        "coord_source": coord_source,
        "coord_shape": list(coords.shape),
        "coords_local_global_shape": list(coords_local_global.shape),
        "coords_local_global_order": "[z_local, x_local, z_global, x_global]",
        "x_coord_min": float(np.min(x_local)),
        "x_coord_max": float(np.max(x_local)),
        "z_coord_min": float(np.min(z_local)),
        "z_coord_max": float(np.max(z_local)),
        "x_global_coord_min": float(np.min(x_global)),
        "x_global_coord_max": float(np.max(x_global)),
        "z_global_coord_min": float(np.min(z_global)),
        "z_global_coord_max": float(np.max(z_global)),
        "x_global_min_mm": x_global_min_mm,
        "x_global_max_mm": x_global_max_mm,
        "z_global_min_mm": z_global_min_mm,
        "z_global_max_mm": z_global_max_mm,
        "x_axis_min_mm": float(np.min(x_axis_mm)),
        "x_axis_max_mm": float(np.max(x_axis_mm)),
        "z_axis_min_mm": float(np.min(z_axis_mm)),
        "z_axis_max_mm": float(np.max(z_axis_mm)),
    }
    return coords, coords_local_global, x_axis_mm.astype(np.float32), z_axis_mm.astype(np.float32), coord_metadata


def _validate_finite_complex(array: np.ndarray, name: str, path: Path) -> None:
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} contains NaN/Inf in {path}")


def _complex_to_2ch_channel_iq(channel_iq_norm: np.ndarray) -> np.ndarray:
    # channel_iq_norm: [T, C] -> x: [2, C, T]
    return np.stack(
        [channel_iq_norm.real.T, channel_iq_norm.imag.T],
        axis=0,
    ).astype(np.float32, copy=False)


def _complex_to_2ch_das_iq(das_iq_norm: np.ndarray) -> np.ndarray:
    # das_iq_norm: [H, W] -> y: [2, H, W]
    return np.stack(
        [das_iq_norm.real, das_iq_norm.imag],
        axis=0,
    ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    sample: Dict[str, Any]
    metadata: Dict[str, Any]


class ToyDASIQDataset(Dataset):
    """Dataset backed by toy Field II DAS-IQ MATLAB samples."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.files = sorted(self.data_dir.glob("sample_*_das.mat"))
        if not self.files:
            raise FileNotFoundError(f"No sample_*_das.mat files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        sample = load_sample_mat(path)

        channel_data = _unwrap_singleton(sample["channel_data"])
        das = _unwrap_singleton(sample["das"])

        channel_iq_norm = _to_complex_array(channel_data["channel_iq_norm"], "channel_data.channel_iq_norm", path)
        das_iq_norm = _to_complex_array(das["das_iq_norm"], "das.das_iq_norm", path)

        _validate_finite_complex(channel_iq_norm, "channel_data.channel_iq_norm", path)
        _validate_finite_complex(das_iq_norm, "das.das_iq_norm", path)

        x = torch.from_numpy(_complex_to_2ch_channel_iq(channel_iq_norm))
        y = torch.from_numpy(_complex_to_2ch_das_iq(das_iq_norm))

        metadata = _extract_metadata(sample, path)
        coords_np, coords_local_global_np, x_axis_mm_np, z_axis_mm_np, coord_metadata = _extract_coord_maps(
            das,
            int(das_iq_norm.shape[0]),
            int(das_iq_norm.shape[1]),
            path,
        )
        metadata.update(coord_metadata)
        coords = torch.from_numpy(coords_np)
        coords_local_global = torch.from_numpy(coords_local_global_np)
        x_axis_mm = torch.from_numpy(x_axis_mm_np)
        z_axis_mm = torch.from_numpy(z_axis_mm_np)

        return {
            "x": x,
            "y": y,
            "coords": coords,
            "coords_local_global": coords_local_global,
            "x_axis_mm": x_axis_mm,
            "z_axis_mm": z_axis_mm,
            "metadata": metadata,
        }


def dbf_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad x in time to Tmax and stack y."""
    if len(batch) == 0:
        raise ValueError("dbf_collate_fn received an empty batch")

    xs = [item["x"] for item in batch]
    ys = [item["y"] for item in batch]
    coords = [item["coords"] for item in batch]
    coords_local_global = [item["coords_local_global"] for item in batch]
    x_axes_mm = [item["x_axis_mm"] for item in batch]
    z_axes_mm = [item["z_axis_mm"] for item in batch]
    metadata = [item["metadata"] for item in batch]

    if any(x.ndim != 3 or x.shape[0] != 2 for x in xs):
        raise ValueError("Expected x tensors with shape [2, C, T]")
    if any(y.ndim != 3 or y.shape[0] != 2 for y in ys):
        raise ValueError("Expected y tensors with shape [2, H, W]")
    if any(coord.ndim != 3 or coord.shape[0] != 2 for coord in coords):
        raise ValueError("Expected coordinate tensors with shape [2, H, W]")
    if any(coord.ndim != 3 or coord.shape[0] != 4 for coord in coords_local_global):
        raise ValueError("Expected local+global coordinate tensors with shape [4, H, W]")

    c_values = {int(x.shape[1]) for x in xs}
    if len(c_values) != 1:
        raise ValueError("All samples in the batch must have the same C dimension")

    h_values = {int(y.shape[1]) for y in ys}
    w_values = {int(y.shape[2]) for y in ys}
    if len(h_values) != 1 or len(w_values) != 1:
        raise ValueError("All samples in the batch must have the same H and W dimensions")

    max_t = max(int(x.shape[2]) for x in xs)
    batch_size = len(batch)
    c = int(xs[0].shape[1])
    h = int(ys[0].shape[1])
    w = int(ys[0].shape[2])

    x_batch = torch.zeros((batch_size, 2, c, max_t), dtype=torch.float32)
    y_batch = torch.stack([y.to(torch.float32) for y in ys], dim=0)
    coords_batch = torch.stack([coord.to(torch.float32) for coord in coords], dim=0)
    coords_local_global_batch = torch.stack([coord.to(torch.float32) for coord in coords_local_global], dim=0)
    x_axis_mm_batch = torch.stack([axis.to(torch.float32) for axis in x_axes_mm], dim=0)
    z_axis_mm_batch = torch.stack([axis.to(torch.float32) for axis in z_axes_mm], dim=0)

    for idx, x in enumerate(xs):
        t = int(x.shape[2])
        x_batch[idx, :, :, :t] = x.to(torch.float32)

    return {
        "x": x_batch,
        "y": y_batch,
        "coords": coords_batch,
        "coords_local_global": coords_local_global_batch,
        "x_axis_mm": x_axis_mm_batch,
        "z_axis_mm": z_axis_mm_batch,
        "metadata": metadata,
    }

