"""Versioned JSON-safe persistence for complete Mode B reconstruction models."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mathfield.core.basis.fourier import DCT2Basis, Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.local.local_patch import LocalPatch
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity

MODEL_CODEC_VERSION = "1.0.0"


class ModeBCodecError(ValueError):
    """Raised when a serialized Mode B model has an invalid or unsupported shape."""


@dataclass(frozen=True)
class _ModeBPatchParts:
    patch_id: int
    bounds: Tuple[float, float, float, float]
    basis_config: Mapping
    basis: Any
    coefficients: np.ndarray
    coefficient_dimensions: int
    error: float


@dataclass(frozen=True)
class _ModeBParts:
    payload: Mapping
    chebyshev_config: Mapping
    fourier_config: Mapping
    chebyshev: Any
    fourier: Any
    include_fourier_dc: bool
    expected_terms: int
    coefficients: np.ndarray
    coefficient_dimensions: int
    channel_count: int
    channel_labels: Tuple[str, ...]
    global_metadata: Mapping
    partition: Mapping
    patches: Tuple[_ModeBPatchParts, ...]


class _ValidatedModeBPayload(dict):
    """A shallow trusted view of a canonical payload validated exactly once.

    Nested canonical values are shared by reference. Internal consumers must not
    mutate this object; plain dictionaries are always treated as untrusted and
    validated again when they cross a public persistence boundary.
    """

    def __init__(self, payload: Mapping, parts: _ModeBParts) -> None:
        super().__init__(payload)
        self._parts = parts

    def __deepcopy__(self, memo: Dict[int, Any]) -> Dict[str, Any]:
        # Explicit caller copies should remain ordinary mutable dictionaries and
        # must not carry the internal trusted marker into a new payload.
        return copy.deepcopy(dict(self), memo)


def _mapping(value: Any, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ModeBCodecError(f"{name} must be an object.")
    return value


def _sequence(value: Any, name: str) -> Sequence:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ModeBCodecError(f"{name} must be an array.")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ModeBCodecError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModeBCodecError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ModeBCodecError(f"{name} must be a finite number.")
    return result


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ModeBCodecError(f"{name} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ModeBCodecError(f"{name} must be at least {minimum}.")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ModeBCodecError(f"{name} must be a boolean.")
    return bool(value)


def _bounds(value: Any, name: str) -> Tuple[float, float, float, float]:
    items = _sequence(value, name)
    if len(items) != 4:
        raise ModeBCodecError(f"{name} must contain exactly four values.")
    result = tuple(_finite_float(item, f"{name}[{index}]") for index, item in enumerate(items))
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ModeBCodecError(f"{name} must satisfy x_min < x_max and y_min < y_max.")
    return result


def _interval(value: Any, name: str) -> Tuple[float, float]:
    items = _sequence(value, name)
    if len(items) != 2:
        raise ModeBCodecError(f"{name} must contain exactly two values.")
    result = (
        _finite_float(items[0], f"{name}[0]"),
        _finite_float(items[1], f"{name}[1]"),
    )
    if result[0] >= result[1]:
        raise ModeBCodecError(f"{name} must satisfy minimum < maximum.")
    return result


def _coefficient_array(value: Any, expected_terms: int, name: str) -> Tuple[np.ndarray, int]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModeBCodecError(f"{name} must be a numeric coefficient array.") from exc
    if array.ndim not in (1, 2):
        raise ModeBCodecError(f"{name} must be a vector or a terms-by-channels matrix.")
    if array.shape[0] != expected_terms:
        raise ModeBCodecError(f"{name} has {array.shape[0]} terms; expected {expected_terms}.")
    if array.ndim == 2 and array.shape[1] < 1:
        raise ModeBCodecError(f"{name} must contain at least one channel.")
    if not np.all(np.isfinite(array)):
        raise ModeBCodecError(f"{name} contains a non-finite coefficient.")
    return array, array.ndim


def _channel_labels(value: Any, channel_count: int) -> Tuple[str, ...]:
    labels = _sequence(value, "channel_labels")
    if len(labels) != channel_count or any(not isinstance(label, str) or not label for label in labels):
        raise ModeBCodecError("channel_labels must contain one non-empty string per coefficient channel.")
    if len(set(labels)) != len(labels):
        raise ModeBCodecError("channel_labels must be unique.")
    return tuple(labels)


def serialize_basis_config(basis: Any) -> Dict[str, Any]:
    """Return a canonical JSON-safe configuration for a supported 2D basis."""
    if isinstance(basis, Mapping):
        basis = deserialize_basis_config(basis)
    if isinstance(basis, Chebyshev2D):
        return {
            "type": "chebyshev2d",
            "deg_x": int(basis.deg_x),
            "deg_y": int(basis.deg_y),
            "truncation": str(basis.truncation),
        }
    if isinstance(basis, Fourier2D):
        return {
            "type": "fourier2d",
            "freq_x": int(basis.freq_x),
            "freq_y": int(basis.freq_y),
            "truncation": str(basis.truncation),
            "frequencies": [[float(fx), float(fy)] for fx, fy in basis.freq_pairs],
        }
    if isinstance(basis, DCT2Basis):
        return {
            "type": "dct2",
            "freq_x": int(basis.freq_x),
            "freq_y": int(basis.freq_y),
            "x_bounds": [float(value) for value in basis.x_bounds],
            "y_bounds": [float(value) for value in basis.y_bounds],
        }
    raise ModeBCodecError(f"Unsupported Mode B basis type: {type(basis).__name__}")


def deserialize_basis_config(data: Any) -> Any:
    """Validate a serialized basis configuration and construct its real basis object."""
    config = _mapping(data, "basis")
    basis_type = config.get("type")
    if basis_type == "chebyshev2d":
        truncation = config.get("truncation")
        if truncation not in ("total_degree", "hyperbolic_cross", "full_tensor", "tensor"):
            raise ModeBCodecError(f"Unsupported Chebyshev truncation: {truncation!r}")
        try:
            return Chebyshev2D(
                deg_x=_integer(config.get("deg_x"), "basis.deg_x"),
                deg_y=_integer(config.get("deg_y"), "basis.deg_y"),
                truncation=truncation,
            )
        except (TypeError, ValueError) as exc:
            raise ModeBCodecError(f"Invalid Chebyshev basis configuration: {exc}") from exc
    if basis_type == "fourier2d":
        truncation = config.get("truncation")
        if truncation not in ("circular", "rectangular"):
            raise ModeBCodecError(f"Unsupported Fourier truncation: {truncation!r}")
        frequencies = None
        if "frequencies" in config:
            raw_frequencies = _sequence(config["frequencies"], "basis.frequencies")
            frequencies = []
            for index, pair in enumerate(raw_frequencies):
                values = _sequence(pair, f"basis.frequencies[{index}]")
                if len(values) != 2:
                    raise ModeBCodecError("Each Fourier frequency must contain exactly two values.")
                frequencies.append(
                    (
                        _finite_float(values[0], f"basis.frequencies[{index}][0]"),
                        _finite_float(values[1], f"basis.frequencies[{index}][1]"),
                    )
                )
        try:
            return Fourier2D(
                frequencies=frequencies,
                freq_x=_integer(config.get("freq_x"), "basis.freq_x"),
                freq_y=_integer(config.get("freq_y"), "basis.freq_y"),
                truncation=truncation,
            )
        except (TypeError, ValueError) as exc:
            raise ModeBCodecError(f"Invalid Fourier basis configuration: {exc}") from exc
    if basis_type == "dct2":
        try:
            return DCT2Basis(
                freq_x=_integer(config.get("freq_x"), "basis.freq_x"),
                freq_y=_integer(config.get("freq_y"), "basis.freq_y"),
                x_bounds=_interval(config.get("x_bounds"), "basis.x_bounds"),
                y_bounds=_interval(config.get("y_bounds"), "basis.y_bounds"),
            )
        except (TypeError, ValueError) as exc:
            raise ModeBCodecError(f"Invalid DCT2 basis configuration: {exc}") from exc
    raise ModeBCodecError(f"Unsupported serialized basis type: {basis_type!r}")


def _partition_settings(local_decomposition: Any, override: Any) -> Dict[str, Any]:
    sources: List[Any] = [override]
    if isinstance(local_decomposition, Mapping):
        sources.extend(
            local_decomposition.get(key)
            for key in ("partition_of_unity", "partition", "pou")
        )
    for source in sources:
        if source is None:
            continue
        if isinstance(source, WendlandPartitionOfUnity):
            overlap = _finite_float(source.overlap_ratio, "partition.overlap_ratio")
            separable = _boolean(source.separable, "partition.separable")
        else:
            config = _mapping(source, "partition_of_unity")
            overlap = _finite_float(config.get("overlap_ratio", 0.33), "partition.overlap_ratio")
            separable = _boolean(config.get("separable", True), "partition.separable")
        if overlap < 0.0:
            raise ModeBCodecError("partition.overlap_ratio must be non-negative.")
        return {"overlap_ratio": overlap, "separable": separable}
    return {"overlap_ratio": 0.33, "separable": True}


def _serialize_patch(
    patch: Any,
    patch_id: int,
    expected_channels: int,
) -> Tuple[Dict[str, Any], _ModeBPatchParts]:
    if isinstance(patch, LocalPatch):
        bounds = patch.bounds
        basis = patch.basis
        coefficients = patch.coefficients
        error = patch.error
    else:
        data = _mapping(patch, f"local_patches[{patch_id}]")
        for key in ("bounds", "basis", "coefficients", "error"):
            if key not in data:
                raise ModeBCodecError(f"local_patches[{patch_id}] missing {key!r}.")
        bounds = data["bounds"]
        basis = data["basis"]
        coefficients = data["coefficients"]
        error = data["error"]

    basis_config = serialize_basis_config(basis)
    basis_object = deserialize_basis_config(basis_config)
    coefficient_array, dimensions = _coefficient_array(
        coefficients,
        basis_object.num_terms,
        f"local_patches[{patch_id}].coefficients",
    )
    if dimensions == 1 and expected_channels > 1:
        # A vector-valued LocalPatch is shared across channels by the composite
        # renderer. Store the broadcast explicitly so every channel is persisted.
        coefficient_array = np.repeat(coefficient_array[:, None], expected_channels, axis=1)
        dimensions = 2
    elif dimensions == 2 and coefficient_array.shape[1] != expected_channels:
        raise ModeBCodecError(
            f"local_patches[{patch_id}] has {coefficient_array.shape[1]} channels; expected {expected_channels}."
        )
    error_value = _finite_float(error, f"local_patches[{patch_id}].error")
    if error_value < 0.0:
        raise ModeBCodecError(f"local_patches[{patch_id}].error must be non-negative.")
    bounds_value = _bounds(bounds, f"local_patches[{patch_id}].bounds")
    serialized = {
        "patch_id": patch_id,
        "bounds": [float(value) for value in bounds_value],
        "basis": basis_config,
        "coefficients": coefficient_array.tolist(),
        "coefficient_dimensions": dimensions,
        "error": error_value,
    }
    return serialized, _ModeBPatchParts(
        patch_id=patch_id,
        bounds=bounds_value,
        basis_config=basis_config,
        basis=basis_object,
        coefficients=coefficient_array,
        coefficient_dimensions=dimensions,
        error=error_value,
    )


def _global_metadata(global_model: Mapping) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in ("type", "formula"):
        if key in global_model and global_model[key] is not None:
            metadata[key] = str(global_model[key])
    for key in ("degree", "fourier_freq", "basis_terms_count", "condition_number", "complexity"):
        if key not in global_model or global_model[key] is None:
            continue
        try:
            value = float(global_model[key])
        except (TypeError, ValueError) as exc:
            raise ModeBCodecError(f"global_model.{key} must be numeric.") from exc
        metadata[key] = value if math.isfinite(value) else None
    return metadata


def _validated_global_metadata(value: Any) -> Mapping:
    metadata = _mapping(value, "global_metadata")
    for key in ("type", "formula"):
        if key in metadata and metadata[key] is not None and not isinstance(metadata[key], str):
            raise ModeBCodecError(f"global_metadata.{key} must be a string.")
    for key in ("degree", "fourier_freq", "basis_terms_count", "condition_number", "complexity"):
        if key not in metadata or metadata[key] is None:
            continue
        _finite_float(metadata[key], f"global_metadata.{key}")
    return metadata


def serialize_mode_b_model(
    global_model: Any,
    local_decomposition: Any = None,
    *,
    channel_labels: Optional[Sequence[str]] = None,
    partition: Any = None,
) -> Dict[str, Any]:
    """Serialize all global and local Mode B state without any rendered raster data."""
    model = _mapping(global_model, "global_model")
    for key in ("chebyshev_basis", "fourier_basis", "coefficients"):
        if key not in model:
            raise ModeBCodecError(f"global_model missing {key!r}.")

    chebyshev_config = serialize_basis_config(model["chebyshev_basis"])
    fourier_config = serialize_basis_config(model["fourier_basis"])
    chebyshev = deserialize_basis_config(chebyshev_config)
    fourier = deserialize_basis_config(fourier_config)
    include_fourier_dc = _boolean(model.get("include_fourier_dc", False), "global_model.include_fourier_dc")
    expected_terms = chebyshev.num_terms + fourier.num_terms - (0 if include_fourier_dc else 1)
    coefficients, dimensions = _coefficient_array(model["coefficients"], expected_terms, "global_model.coefficients")
    channel_count = 1 if dimensions == 1 else coefficients.shape[1]

    if channel_labels is None:
        channel_labels = model.get("channel_labels")
    if channel_labels is None:
        defaults = {1: ["Y"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}
        channel_labels = defaults.get(channel_count, [f"channel_{index}" for index in range(channel_count)])
    labels = _channel_labels(channel_labels, channel_count)

    if isinstance(local_decomposition, Mapping):
        patches = local_decomposition.get("patches", [])
    elif local_decomposition is None:
        patches = []
    else:
        patches = local_decomposition
    patch_items = _sequence(patches, "local_patches")
    serialized_patches: List[Dict[str, Any]] = []
    patch_parts: List[_ModeBPatchParts] = []
    for index, patch in enumerate(patch_items):
        serialized, parts = _serialize_patch(patch, index, channel_count)
        serialized_patches.append(serialized)
        patch_parts.append(parts)

    metadata = _global_metadata(model)
    partition_data = _partition_settings(local_decomposition, partition)
    payload: Dict[str, Any] = {
        "codec_version": MODEL_CODEC_VERSION,
        "global_basis": {
            "type": "hybrid_chebyshev_fourier",
            "chebyshev": chebyshev_config,
            "fourier": fourier_config,
            "include_fourier_dc": include_fourier_dc,
        },
        "coefficients": coefficients.tolist(),
        "coefficient_dimensions": dimensions,
        "channel_labels": list(labels),
        "global_metadata": metadata,
        "partition_of_unity": partition_data,
        "local_patches": serialized_patches,
    }
    parts = _ModeBParts(
        payload=payload,
        chebyshev_config=chebyshev_config,
        fourier_config=fourier_config,
        chebyshev=chebyshev,
        fourier=fourier,
        include_fourier_dc=include_fourier_dc,
        expected_terms=expected_terms,
        coefficients=coefficients,
        coefficient_dimensions=dimensions,
        channel_count=channel_count,
        channel_labels=labels,
        global_metadata=metadata,
        partition=partition_data,
        patches=tuple(patch_parts),
    )
    return _ValidatedModeBPayload(payload, parts)


def _parse_mode_b_payload(data: Any) -> _ModeBParts:
    payload = _mapping(data, "mode_b_model")
    if payload.get("codec_version") != MODEL_CODEC_VERSION:
        raise ModeBCodecError(
            f"Unsupported Mode B codec version: {payload.get('codec_version')!r} (expected {MODEL_CODEC_VERSION})"
        )

    global_basis = _mapping(payload.get("global_basis"), "global_basis")
    if global_basis.get("type") != "hybrid_chebyshev_fourier":
        raise ModeBCodecError("global_basis.type must be 'hybrid_chebyshev_fourier'.")
    chebyshev_config = _mapping(global_basis.get("chebyshev"), "global_basis.chebyshev")
    fourier_config = _mapping(global_basis.get("fourier"), "global_basis.fourier")
    chebyshev = deserialize_basis_config(chebyshev_config)
    fourier = deserialize_basis_config(fourier_config)
    include_fourier_dc = _boolean(global_basis.get("include_fourier_dc"), "global_basis.include_fourier_dc")
    expected_terms = chebyshev.num_terms + fourier.num_terms - (0 if include_fourier_dc else 1)
    coefficients, dimensions = _coefficient_array(payload.get("coefficients"), expected_terms, "coefficients")
    stored_dimensions = _integer(payload.get("coefficient_dimensions"), "coefficient_dimensions", 1)
    if stored_dimensions != dimensions:
        raise ModeBCodecError("coefficient_dimensions does not match the coefficient array shape.")
    channel_count = 1 if dimensions == 1 else coefficients.shape[1]
    labels = _channel_labels(payload.get("channel_labels"), channel_count)
    metadata = _validated_global_metadata(payload.get("global_metadata", {}))

    partition = _mapping(payload.get("partition_of_unity"), "partition_of_unity")
    partition_data = _partition_settings({"partition_of_unity": partition}, None)
    raw_patches = _sequence(payload.get("local_patches"), "local_patches")
    patches: List[_ModeBPatchParts] = []
    for index, raw_patch in enumerate(raw_patches):
        patch = _mapping(raw_patch, f"local_patches[{index}]")
        patch_id = _integer(patch.get("patch_id", index), f"local_patches[{index}].patch_id")
        if patch_id != index:
            raise ModeBCodecError("local patch IDs must be contiguous and match their array order.")
        basis_config = _mapping(patch.get("basis"), f"local_patches[{index}].basis")
        basis = deserialize_basis_config(basis_config)
        patch_coefficients, patch_dimensions = _coefficient_array(
            patch.get("coefficients"),
            basis.num_terms,
            f"local_patches[{index}].coefficients",
        )
        stored_patch_dimensions = _integer(
            patch.get("coefficient_dimensions"),
            f"local_patches[{index}].coefficient_dimensions",
            1,
        )
        if stored_patch_dimensions != patch_dimensions:
            raise ModeBCodecError(f"local_patches[{index}].coefficient_dimensions does not match its shape.")
        if patch_dimensions == 1 and channel_count > 1:
            patch_coefficients = np.repeat(patch_coefficients[:, None], channel_count, axis=1)
            patch_dimensions = 2
        elif patch_dimensions == 2 and patch_coefficients.shape[1] != channel_count:
            raise ModeBCodecError(f"local_patches[{index}] channel count does not match the global model.")
        error = _finite_float(patch.get("error"), f"local_patches[{index}].error")
        if error < 0.0:
            raise ModeBCodecError(f"local_patches[{index}].error must be non-negative.")
        patches.append(
            _ModeBPatchParts(
                patch_id=patch_id,
                bounds=_bounds(patch.get("bounds"), f"local_patches[{index}].bounds"),
                basis_config=basis_config,
                basis=basis,
                coefficients=patch_coefficients,
                coefficient_dimensions=patch_dimensions,
                error=error,
            )
        )

    return _ModeBParts(
        payload=payload,
        chebyshev_config=chebyshev_config,
        fourier_config=fourier_config,
        chebyshev=chebyshev,
        fourier=fourier,
        include_fourier_dc=include_fourier_dc,
        expected_terms=expected_terms,
        coefficients=coefficients,
        coefficient_dimensions=dimensions,
        channel_count=channel_count,
        channel_labels=labels,
        global_metadata=metadata,
        partition=partition_data,
        patches=tuple(patches),
    )


def _validated_mode_b_payload(data: Any) -> _ValidatedModeBPayload:
    if isinstance(data, _ValidatedModeBPayload):
        return data
    parts = _parse_mode_b_payload(data)
    return _ValidatedModeBPayload(parts.payload, parts)


def deserialize_mode_b_model(data: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate a codec payload and restore real basis and LocalPatch objects."""
    trusted = _validated_mode_b_payload(data)
    parts = trusted._parts
    chebyshev = deserialize_basis_config(parts.chebyshev_config)
    fourier = deserialize_basis_config(parts.fourier_config)
    global_model = dict(parts.global_metadata)
    global_model["type"] = parts.global_metadata.get("type", "mode_b_global_hybrid")
    global_model["degree"] = int(chebyshev.deg_x)
    global_model["fourier_freq"] = int(fourier.freq_x)
    global_model["basis_terms_count"] = parts.expected_terms
    global_model["include_fourier_dc"] = parts.include_fourier_dc
    global_model["chebyshev_basis"] = chebyshev
    global_model["fourier_basis"] = fourier
    global_model["coefficients"] = np.array(parts.coefficients, dtype=np.float64, copy=True)
    global_model["channel_labels"] = list(parts.channel_labels)

    patches: List[LocalPatch] = []
    for patch in parts.patches:
        patches.append(
            LocalPatch(
                bounds=patch.bounds,
                basis=deserialize_basis_config(patch.basis_config),
                coefficients=np.array(patch.coefficients, dtype=np.float64, copy=True),
                error=patch.error,
            )
        )

    local_decomposition = {
        "patches": patches,
        "leaves_count": len(patches),
        "partition_of_unity": dict(parts.partition),
    }
    return global_model, local_decomposition


def validate_mode_b_model(data: Any) -> None:
    """Validate a codec payload at a persistence boundary without rebuilding the model."""
    _validated_mode_b_payload(data)


def is_mode_b_global_model(model: Any) -> bool:
    """Return whether a model dictionary carries the Mode B hybrid representation."""
    if not isinstance(model, Mapping):
        return False
    model_type = str(model.get("type", ""))
    if model_type.startswith("mode_b"):
        return True
    return all(key in model for key in ("chebyshev_basis", "fourier_basis", "coefficients"))


def mode_b_payload_from_result(result: Any) -> Optional[Dict[str, Any]]:
    """Extract or build a canonical Mode B payload from an export/context-style dictionary."""
    if not isinstance(result, Mapping):
        return None
    if result.get("mode_b_model") is not None:
        return _validated_mode_b_payload(result["mode_b_model"])

    global_model = result.get("global_model")
    if not is_mode_b_global_model(global_model):
        return None
    local_decomposition = result.get("local_decomposition")
    if not isinstance(local_decomposition, Mapping):
        local_decomposition = {"patches": result.get("local_patches") or []}
    partition = result.get("partition_of_unity", result.get("partition"))
    channel_labels = result.get("channel_labels")
    if channel_labels is None:
        color_space = str(result.get("color_space", "")).upper()
        if color_space == "RGBA":
            channel_labels = list("RGBA")
        elif color_space == "RGB":
            channel_labels = list("RGB")
    return serialize_mode_b_model(
        global_model,
        local_decomposition,
        channel_labels=channel_labels,
        partition=partition,
    )


def _mode_b_parameter_table(parts: _ModeBParts) -> Dict[str, Any]:
    coefficients = parts.coefficients
    matrix = coefficients.reshape(-1, 1) if coefficients.ndim == 1 else coefficients
    labels = parts.channel_labels
    parameters: Dict[str, Any] = {}
    for channel_index, label in enumerate(labels):
        for term_index, value in enumerate(matrix[:, channel_index]):
            parameters[f"{label}_c{term_index}"] = {
                "stage": "global",
                "channel": label,
                "value": float(value),
            }
    for patch_index, patch in enumerate(parts.patches):
        patch_coefficients = patch.coefficients
        patch_matrix = patch_coefficients.reshape(-1, 1) if patch_coefficients.ndim == 1 else patch_coefficients
        bounds = patch.bounds
        domain = f"[{bounds[0]:.17g}, {bounds[2]:.17g}]x[{bounds[1]:.17g}, {bounds[3]:.17g}]"
        for channel_index, label in enumerate(labels):
            for term_index, value in enumerate(patch_matrix[:, channel_index]):
                parameters[f"patch_{patch_index}_{label}_c{term_index}"] = {
                    "stage": "local",
                    "channel": label,
                    "value": float(value),
                    "domain": domain,
                    "patch_id": patch_index,
                }
    return parameters


def mode_b_parameter_table(payload: Any) -> Dict[str, Any]:
    """Build named global and per-patch coefficient records for every channel."""
    return _mode_b_parameter_table(_validated_mode_b_payload(payload)._parts)


def _chebyshev_expression(order: int, variable: str) -> str:
    if order <= 0:
        return "1"
    if order == 1:
        return variable
    previous, current = "1", variable
    for _ in range(2, order + 1):
        previous, current = current, f"(2*{variable}*{current} - {previous})"
    return current


def _mode_b_machine_formula(parts: _ModeBParts, channel_index: int) -> str:
    coefficients = parts.coefficients
    channel_coefficients = coefficients[:, channel_index] if coefficients.ndim == 2 else coefficients
    terms: List[Tuple[float, str]] = []
    for term_index, (degree_x, degree_y) in enumerate(parts.chebyshev.indices):
        value = float(channel_coefficients[term_index])
        if abs(value) <= 0.0:
            continue
        if degree_x == 0 and degree_y == 0:
            name = "1"
        elif degree_y == 0:
            name = f"({_chebyshev_expression(int(degree_x), 'x')})"
        elif degree_x == 0:
            name = f"({_chebyshev_expression(int(degree_y), 'y')})"
        else:
            name = (
                f"({_chebyshev_expression(int(degree_x), 'x')})"
                f"*({_chebyshev_expression(int(degree_y), 'y')})"
            )
        terms.append((abs(value), f"{value:+.17g}*{name}"))

    base = parts.chebyshev.num_terms
    for harmonic_index, (frequency_x, frequency_y) in enumerate(parts.fourier.freq_pairs):
        cosine = float(channel_coefficients[base + 2 * harmonic_index])
        sine = float(channel_coefficients[base + 2 * harmonic_index + 1])
        argument = f"({int(frequency_x)}*pi*x" if frequency_x else "("
        if frequency_y:
            argument += f" + {int(frequency_y)}*pi*y)" if frequency_x else f"{int(frequency_y)}*pi*y)"
        else:
            argument += ")"
        for value, function in ((cosine, "cos"), (sine, "sin")):
            if abs(value) > 0.0:
                terms.append((abs(value), f"{value:+.17g}*{function}{argument}"))

    terms.sort(key=lambda item: item[0], reverse=True)
    return " ".join(text for _, text in terms) or "0.0"


def mode_b_machine_formulas(payload: Any) -> Dict[str, str]:
    """Regenerate machine channel equations from canonical coefficients."""
    parts = _validated_mode_b_payload(payload)._parts
    return {
        label: _mode_b_machine_formula(parts, channel_index)
        for channel_index, label in enumerate(parts.channel_labels)
    }


def _mode_b_channel_records(
    parts: _ModeBParts,
    formulas: Optional[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    coefficients = parts.coefficients
    matrix = coefficients.reshape(-1, 1) if coefficients.ndim == 1 else coefficients
    formulas = formulas or {}
    records: List[Dict[str, Any]] = []
    for channel_index, label in enumerate(parts.channel_labels):
        records.append(
            {
                "channel": label,
                "formula": formulas.get(label, f"hybrid_chebyshev_fourier_channel_{label}"),
                "parameters": [float(value) for value in matrix[:, channel_index]],
            }
        )
    return records


def mode_b_channel_records(payload: Any, formulas: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    """Build complete per-channel project records from a Mode B payload."""
    if formulas is not None and not isinstance(formulas, Mapping):
        raise ModeBCodecError("formulas must be an object keyed by channel label.")
    return _mode_b_channel_records(_validated_mode_b_payload(payload)._parts, formulas)


def mode_b_channel_formulas(payload: Any, formulas: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Return deterministic per-channel formula metadata without copying coefficients."""
    if formulas is not None and not isinstance(formulas, Mapping):
        raise ModeBCodecError("formulas must be an object keyed by channel label.")
    parts = _validated_mode_b_payload(payload)._parts
    formulas = formulas or {}
    channel_formulas: Dict[str, str] = {}
    for label in parts.channel_labels:
        value = formulas.get(label)
        channel_formulas[label] = (
            f"hybrid_chebyshev_fourier_channel_{label}" if value is None else str(value)
        )
    return channel_formulas


def mode_b_legacy_views(
    payload: Any,
    formulas: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Derive compatibility views only when a consumer explicitly needs them."""
    if formulas is not None and not isinstance(formulas, Mapping):
        raise ModeBCodecError("formulas must be an object keyed by channel label.")
    trusted = _validated_mode_b_payload(payload)
    parts = trusted._parts
    global_model = dict(parts.global_metadata)
    global_model["type"] = parts.global_metadata.get("type", "mode_b_global_hybrid")
    global_model["degree"] = int(parts.chebyshev.deg_x)
    global_model["fourier_freq"] = int(parts.fourier.freq_x)
    global_model["basis_terms_count"] = parts.expected_terms
    global_model["include_fourier_dc"] = parts.include_fourier_dc
    global_model["chebyshev_basis"] = parts.chebyshev_config
    global_model["fourier_basis"] = parts.fourier_config
    global_model["coefficients"] = parts.payload["coefficients"]
    global_model["channel_labels"] = list(parts.channel_labels)
    global_model["channels"] = _mode_b_channel_records(parts, formulas)
    global_model["local_patches_count"] = len(parts.patches)
    local_patches = parts.payload["local_patches"]
    return {
        "global_model": global_model,
        "parameters": _mode_b_parameter_table(parts),
        "local_patches": local_patches,
        "local_decomposition": {
            "patches": local_patches,
            "leaves_count": len(parts.patches),
            "partition_of_unity": dict(parts.partition),
        },
        "basis_terms_count": parts.expected_terms,
    }
