"""
mathfield/export/project_file.py

Versioned JSON Project (.mathfield) serialization and schema validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


class ProjectFileError(Exception):
    """Raised when a .mathfield project file fails schema validation or corruption check."""



REQUIRED_ROOT_KEYS = [
    "schema_version",
    "coordinate_domain",
    "color_space",
    "global_model",
    "final_metrics",
]


def validate_project_dict(data: Dict[str, Any]) -> None:
    """Validate that dictionary conforms to MathField Project Schema v1.0.0."""
    if not isinstance(data, dict):
        raise ProjectFileError("Project root must be a JSON object.")

    for k in REQUIRED_ROOT_KEYS:
        if k not in data:
            raise ProjectFileError(f"Missing required project property: '{k}'")

    domain = data["coordinate_domain"]
    if not isinstance(domain, dict):
        raise ProjectFileError("'coordinate_domain' must be an object.")
    for dk in ["x_min", "x_max", "y_min", "y_max"]:
        if dk not in domain:
            raise ProjectFileError(f"'coordinate_domain' missing '{dk}'")

    model = data["global_model"]
    if not isinstance(model, dict):
        raise ProjectFileError("'global_model' must be an object.")
    if "channels" not in model or not isinstance(model["channels"], list):
        raise ProjectFileError("'global_model' must contain a 'channels' array.")

    metrics = data["final_metrics"]
    if not isinstance(metrics, dict):
        raise ProjectFileError("'final_metrics' must be an object.")
    if "mse" not in metrics and "psnr" not in metrics:
        raise ProjectFileError("'final_metrics' missing 'mse'")
    if "mse" not in metrics and "psnr" in metrics:
        metrics["mse"] = 10.0 ** (-float(metrics["psnr"]) / 10.0)
    if "psnr" not in metrics and "mse" in metrics:
        mse_val = max(float(metrics["mse"]), 1e-15)
        metrics["psnr"] = -10.0 * math.log10(mse_val)


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


def save_project(
    project_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """Save project dictionary to a .mathfield JSON file."""
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    # Work on a shallow copy so validation-derived metrics never mutate the caller.
    payload = dict(project_data)
    if isinstance(project_data.get("final_metrics"), dict):
        payload["final_metrics"] = dict(project_data["final_metrics"])
    validate_project_dict(payload)

    content = json.dumps(payload, default=_json_fallback, indent=2, ensure_ascii=False)
    out_p.write_text(content, encoding="utf-8")


def load_project(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate a .mathfield JSON project file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")

    try:
        content = p.read_text(encoding="utf-8-sig")
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ProjectFileError(f"Corrupt JSON in project file: {exc}") from exc

    validate_project_dict(data)
    return data
