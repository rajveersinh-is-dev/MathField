"""
mathfield/export/csv_export.py

High-precision CSV parameter table exporter.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Union


def export_csv_parameters(
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> None:
    """
    Exports discovered parameters into a structured CSV table.
    Preserves IEEE-754 precision (>= 15 significant digits).
    """
    out_p = Path(output_path)
    if not out_p.parent.exists():
        raise FileNotFoundError(f"Destination directory does not exist: {out_p.parent}")

    raw_params = result.get("parameters", {})
    rows: List[Dict[str, Any]] = []

    if isinstance(raw_params, dict):
        for k, v in raw_params.items():
            if isinstance(v, dict):
                val = float(v.get("value", 0.0))
                stage = str(v.get("stage", "global"))
                channel = str(v.get("channel", "RGB"))
                domain = str(v.get("domain", "[-1, 1]x[-1, 1]"))
                comp = str(v.get("complexity", "1.0"))
            else:
                val = float(v)
                stage = "global"
                channel = "RGB"
                domain = "[-1, 1]x[-1, 1]"
                comp = "1.0"

            if not math.isfinite(val):
                val = 0.0

            rows.append(
                {
                    "parameter_name": str(k),
                    "parameter_value": f"{val:.15g}",
                    "stage": stage,
                    "channel": channel,
                    "domain": domain,
                    "complexity": comp,
                }
            )
    elif isinstance(raw_params, (list, tuple)):
        for i, v in enumerate(raw_params):
            val = float(v)
            if not math.isfinite(val):
                val = 0.0
            rows.append(
                {
                    "parameter_name": f"c{i}",
                    "parameter_value": f"{val:.15g}",
                    "stage": "global",
                    "channel": "RGB",
                    "domain": "[-1, 1]x[-1, 1]",
                    "complexity": "1.0",
                }
            )

    if not rows:
        # Fallback single parameter
        rows.append(
            {
                "parameter_name": "c0",
                "parameter_value": "1.000000000000000",
                "stage": "global",
                "channel": "RGB",
                "domain": "[-1, 1]x[-1, 1]",
                "complexity": "1.0",
            }
        )

    fieldnames = [
        "parameter_name",
        "parameter_value",
        "stage",
        "channel",
        "domain",
        "complexity",
    ]

    with out_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
