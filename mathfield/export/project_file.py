"""
mathfield/export/project_file.py

Versioned JSON Project (.mathfield) serialization and schema validation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from mathfield.core.model_codec import (
    mode_b_channel_formulas,
    mode_b_legacy_views,
    mode_b_machine_formulas,
    mode_b_payload_from_result,
)


class ProjectFileError(Exception):
    """Raised when a .mathfield project file fails schema validation or corruption check."""


PROJECT_SCHEMA_VERSION = "1.0.0"
COMPACT_PROJECT_SCHEMA_VERSION = "1.1.0"
SUPPORTED_PROJECT_SCHEMA_VERSIONS = {PROJECT_SCHEMA_VERSION, COMPACT_PROJECT_SCHEMA_VERSION}

REQUIRED_ROOT_KEYS = [
    "schema_version",
    "coordinate_domain",
    "color_space",
    "global_model",
    "final_metrics",
]
COMPACT_REQUIRED_ROOT_KEYS = [
    "schema_version",
    "coordinate_domain",
    "color_space",
    "mode_b_model",
    "final_metrics",
]
LEGACY_ONLY_COMPACT_KEYS = (
    "local_patches",
    "parameters",
    "partition_of_unity",
)


def _validate_finite_json(value: Any, name: str = "project") -> None:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ProjectFileError(f"{name} contains a non-finite numeric value.")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.floating) and not np.all(np.isfinite(value)):
            raise ProjectFileError(f"{name} contains a non-finite numeric value.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite_json(item, f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{name}[{index}]")


def _project_domain(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectFileError("'coordinate_domain' must be an object.")
    domain = dict(value)
    for key in ("x_min", "x_max", "y_min", "y_max"):
        if key not in domain:
            raise ProjectFileError(f"'coordinate_domain' missing '{key}'")
        raw = domain[key]
        if isinstance(raw, (bool, np.bool_)):
            raise ProjectFileError(f"'coordinate_domain.{key}' must be a finite number.")
        try:
            domain[key] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ProjectFileError(f"'coordinate_domain.{key}' must be a finite number.") from exc
        if not math.isfinite(domain[key]):
            raise ProjectFileError(f"'coordinate_domain.{key}' must be a finite number.")
    if domain["x_min"] >= domain["x_max"] or domain["y_min"] >= domain["y_max"]:
        raise ProjectFileError("'coordinate_domain' must satisfy x_min < x_max and y_min < y_max.")
    return domain


def _project_metrics(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ProjectFileError("'final_metrics' must be an object.")
    metrics = dict(value)
    if "mse" not in metrics and "psnr" not in metrics:
        raise ProjectFileError("'final_metrics' missing 'mse'")

    for key in ("mse", "psnr"):
        if key not in metrics:
            continue
        raw = metrics[key]
        if isinstance(raw, (bool, np.bool_)):
            raise ProjectFileError(f"'final_metrics.{key}' must be a finite number.")
        try:
            metrics[key] = float(raw)
        except (TypeError, ValueError) as exc:
            raise ProjectFileError(f"'final_metrics.{key}' must be a finite number.") from exc
        if not math.isfinite(metrics[key]):
            raise ProjectFileError(f"'final_metrics.{key}' must be a finite number.")

    if "mse" not in metrics and "psnr" in metrics:
        metrics["mse"] = 10.0 ** (-metrics["psnr"] / 10.0)
    if "psnr" not in metrics and "mse" in metrics:
        mse_val = max(metrics["mse"], 1e-15)
        metrics["psnr"] = -10.0 * math.log10(mse_val)
    return metrics


def _formulas_from_global_model(global_model: Any) -> Dict[str, Any]:
    if not isinstance(global_model, Mapping):
        return {}
    channels = global_model.get("channels")
    if not isinstance(channels, list):
        return {}
    formulas: Dict[str, Any] = {}
    for record in channels:
        if not isinstance(record, Mapping):
            continue
        label = record.get("channel")
        formula = record.get("formula")
        if isinstance(label, str) and formula is not None:
            formulas[label] = formula
    return formulas


def _project_channel_formulas(data: Mapping, payload: Dict[str, Any]) -> Dict[str, str]:
    formula_source = data.get("channel_formula_source", "provided")
    if formula_source not in ("provided", "mode_b_model"):
        raise ProjectFileError("'channel_formula_source' must be 'provided' or 'mode_b_model'.")
    raw_formulas = data.get("channel_formulas")
    if raw_formulas is None:
        raw_formulas = _formulas_from_global_model(data.get("global_model"))
    if not isinstance(raw_formulas, Mapping):
        raise ProjectFileError("'channel_formulas' must be an object keyed by channel label.")
    labels = set(payload["channel_labels"])
    unknown = set(raw_formulas) - labels
    if unknown:
        raise ProjectFileError(f"'channel_formulas' contains unknown channels: {sorted(unknown)!r}")
    for label, formula in raw_formulas.items():
        if not isinstance(label, str) or not isinstance(formula, str):
            raise ProjectFileError("'channel_formulas' values must be strings keyed by channel label.")
    formulas = mode_b_channel_formulas(payload, raw_formulas)
    if formula_source == "mode_b_model":
        expected = mode_b_channel_formulas(payload)
        if formulas != expected:
            raise ProjectFileError("Generated channel formulas must use their canonical descriptors.")
        return expected
    return formulas


def _validate_project_data(data: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not isinstance(data, dict):
        raise ProjectFileError("Project root must be a JSON object.")
    _validate_finite_json(data)

    schema_version = data.get("schema_version")
    if schema_version not in SUPPORTED_PROJECT_SCHEMA_VERSIONS:
        raise ProjectFileError(f"Unsupported project schema version: {schema_version!r}")

    required_keys = REQUIRED_ROOT_KEYS if schema_version == PROJECT_SCHEMA_VERSION else COMPACT_REQUIRED_ROOT_KEYS
    for key in required_keys:
        if key not in data:
            raise ProjectFileError(f"Missing required project property: '{key}'")

    payload = dict(data)
    payload["coordinate_domain"] = _project_domain(data["coordinate_domain"])
    payload["final_metrics"] = _project_metrics(data["final_metrics"])
    if not isinstance(data["color_space"], str) or not data["color_space"]:
        raise ProjectFileError("'color_space' must be a non-empty string.")

    trusted_mode_b: Optional[Dict[str, Any]] = None
    if schema_version == PROJECT_SCHEMA_VERSION:
        model = data["global_model"]
        if not isinstance(model, dict):
            raise ProjectFileError("'global_model' must be an object.")
        if "channels" not in model or not isinstance(model["channels"], list):
            raise ProjectFileError("'global_model' must contain a 'channels' array.")
        if data.get("mode_b_model") is not None:
            try:
                trusted_mode_b = mode_b_payload_from_result({"mode_b_model": data["mode_b_model"]})
            except ValueError as exc:
                raise ProjectFileError(f"Invalid Mode B model payload: {exc}") from exc
            if trusted_mode_b is None:
                raise ProjectFileError("Invalid Mode B model payload.")
    else:
        if data.get("mode_b_model") is None:
            raise ProjectFileError("Compact Mode B projects require 'mode_b_model'.")
        try:
            trusted_mode_b = mode_b_payload_from_result({"mode_b_model": data["mode_b_model"]})
        except ValueError as exc:
            raise ProjectFileError(f"Invalid Mode B model payload: {exc}") from exc
        if trusted_mode_b is None:
            raise ProjectFileError("Invalid Mode B model payload.")
        payload["channel_formulas"] = _project_channel_formulas(data, trusted_mode_b)

    return payload, trusted_mode_b


def validate_project_dict(data: Dict[str, Any]) -> None:
    """Validate a MathField project using the legacy v1.0 or compact v1.1 schema."""
    _validate_project_data(data)


def _compact_project_payload(data: Mapping, mode_b_payload: Dict[str, Any]) -> Dict[str, Any]:
    compact = dict(data)
    formulas = _project_channel_formulas(data, mode_b_payload)
    formula_source = data.get("channel_formula_source", "provided")
    if formula_source == "provided" and formulas == mode_b_machine_formulas(mode_b_payload):
        formulas = mode_b_channel_formulas(mode_b_payload)
        formula_source = "mode_b_model"
    compact["channel_formulas"] = formulas
    compact["channel_formula_source"] = formula_source
    compact["mode_b_model"] = mode_b_payload

    metadata = mode_b_payload.get("global_metadata") or {}
    global_basis = mode_b_payload["global_basis"]
    term_count = len(mode_b_payload["coefficients"])
    global_summary = {
        "type": metadata.get("type", "mode_b_global_hybrid"),
        "degree": int(global_basis["chebyshev"]["deg_x"]),
        "fourier_freq": int(global_basis["fourier"]["freq_x"]),
        "basis_terms_count": term_count,
        "coefficients_ref": "mode_b_model.coefficients",
        "channels": [
            {"channel": label, "formula": formulas[label]}
            for label in mode_b_payload["channel_labels"]
        ],
    }
    if metadata.get("condition_number") is not None:
        global_summary["condition_number"] = metadata["condition_number"]
    compact["global_model"] = global_summary
    compact["basis_terms_count"] = term_count
    compact["local_decomposition"] = {
        "leaves_count": len(mode_b_payload["local_patches"]),
        "partition_of_unity": dict(mode_b_payload["partition_of_unity"]),
        "coefficients_ref": "mode_b_model.local_patches",
        "patches": [
            {
                "patch_id": patch["patch_id"],
                "bounds": patch["bounds"],
                "basis": patch["basis"],
                "error": patch["error"],
            }
            for patch in mode_b_payload["local_patches"]
        ],
    }
    for key in LEGACY_ONLY_COMPACT_KEYS:
        compact.pop(key, None)
    return compact


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def _reject_nonfinite_json(token: str) -> None:
    raise ProjectFileError(f"Non-finite JSON number {token!r} is not allowed")


def save_project(
    project_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """Save a validated v1.0 legacy or v1.1 compact project dictionary."""
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    # Validation and metric normalization operate on copies and never mutate the
    # caller. A trusted Mode B payload is reused without another validation pass.
    payload, trusted_mode_b = _validate_project_data(project_data)
    if payload["schema_version"] == COMPACT_PROJECT_SCHEMA_VERSION:
        if trusted_mode_b is None:
            raise ProjectFileError("Compact Mode B projects require a validated model payload.")
        payload = _compact_project_payload(payload, trusted_mode_b)

    try:
        content = json.dumps(payload, default=_json_fallback, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProjectFileError("Project contains a non-finite or unsupported numeric value") from exc
    out_p.write_text(content, encoding="utf-8")


def load_project(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a project and derive legacy views for compact Mode B files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")

    try:
        content = p.read_text(encoding="utf-8-sig")
        data = json.loads(content, parse_constant=_reject_nonfinite_json)
    except json.JSONDecodeError as exc:
        raise ProjectFileError(f"Corrupt JSON in project file: {exc}") from exc

    payload, trusted_mode_b = _validate_project_data(data)
    if trusted_mode_b is not None:
        payload["mode_b_model"] = trusted_mode_b
    if payload["schema_version"] == COMPACT_PROJECT_SCHEMA_VERSION:
        if trusted_mode_b is None:
            raise ProjectFileError("Compact Mode B projects require a validated model payload.")
        formulas = payload["channel_formulas"]
        if payload.get("channel_formula_source", "provided") == "mode_b_model":
            formulas = mode_b_machine_formulas(trusted_mode_b)
        payload.update(mode_b_legacy_views(trusted_mode_b, formulas))
    return payload


def build_mode_b_project(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the shared deterministic schema-1.1 project used by CLI and bundles."""
    if not isinstance(result, Mapping):
        raise ProjectFileError("Mode B project input must be an object.")
    mode_b_payload = mode_b_payload_from_result(result)
    if mode_b_payload is None:
        raise ProjectFileError("Mode B project input does not contain a Mode B model.")

    formulas = result.get("expressions")
    if not isinstance(formulas, Mapping):
        formulas = _formulas_from_global_model(result.get("global_model"))
    metrics = result.get("metrics", result.get("final_metrics", {}))
    if not isinstance(metrics, Mapping):
        raise ProjectFileError("Mode B project metrics must be an object.")
    domain = result.get("coordinate_domain")
    if not domain:
        domain = {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "pixel_width": 64,
            "pixel_height": 64,
            "aspect_ratio": 1.0,
        }

    provided_formulas = mode_b_channel_formulas(mode_b_payload, formulas)
    generated_formulas = mode_b_machine_formulas(mode_b_payload)
    if provided_formulas == generated_formulas:
        channel_formulas = mode_b_channel_formulas(mode_b_payload)
        formula_source = "mode_b_model"
    else:
        channel_formulas = provided_formulas
        formula_source = "provided"
    project: Dict[str, Any] = {
        "schema_version": COMPACT_PROJECT_SCHEMA_VERSION,
        "project_name": str(result.get("project_name", "MathField")),
        "coordinate_domain": dict(domain),
        "color_space": str(result.get("color_space", "RGB")),
        "mode": str(result.get("mode", "MODE_B_FIDELITY")),
        "mode_b_model": mode_b_payload,
        "channel_formulas": channel_formulas,
        "channel_formula_source": formula_source,
        "final_metrics": dict(metrics),
        "random_seed": int(result.get("random_seed", 42)),
    }
    for key in ("condition_report", "status"):
        if key in result and result[key] is not None:
            project[key] = result[key]
    return project
