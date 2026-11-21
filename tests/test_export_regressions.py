"""Regression tests for exporters, CLI honesty and portability."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.ablations import get_ablation_config
from cli import main as cli_main
from examples.generate_sample import build_sample
from mathfield.export.bundle import export_bundle
from mathfield.export.raster import _model_to_image, natural_key, to_uint8_array
from mathfield.export.svg import export_svg


def test_natural_parameter_sort() -> None:
    assert sorted(["c10", "c2", "c1"], key=natural_key) == ["c1", "c2", "c10"]


def test_negative_unit_range_is_remapped_not_clipped() -> None:
    arr = np.array([[-1.0, -0.5], [0.5, 1.0]])
    np.testing.assert_array_equal(to_uint8_array(arr), np.array([[0, 63], [191, 255]], dtype=np.uint8))
    model = {"expressions": {"R": "-1", "G": "-1", "B": "1"}}
    image = _model_to_image(model, width=2, height=2)
    assert np.asarray(image)[0, 0, 0] == 0


def test_svg_embeds_real_reconstruction(tmp_path: Path) -> None:
    model = {"expressions": {"R": "x", "G": "y", "B": "0"}}
    out = tmp_path / "real.svg"
    export_svg(model, out, width=32, height=24)
    text = out.read_text(encoding="utf-8")
    assert "data:image/png;base64," in text
    assert "#0a0e14" not in text  # old decorative background removed


def test_bundle_zip_is_deterministic(tmp_path: Path) -> None:
    model = {"expressions": {"R": "x", "G": "y", "B": "0"}, "parameters": {"c0": 1.0}}
    first = tmp_path / "a.zip"
    second = tmp_path / "b.zip"
    export_bundle(model, first, as_zip=True)
    export_bundle(model, second, as_zip=True)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {
            "reproduce.py",
            "verify.py",
            "project.mathfield",
            "parameters.csv",
            "report.pdf",
            "report.tex",
            "README.md",
        }


def test_bundle_verify_checks_exported_model(tmp_path: Path) -> None:
    rendered = np.zeros((48, 64, 3), dtype=np.float64)
    rendered[:] = [0.5, 0.25, 0.1]
    model = {
        "expressions": {"R": "0.5", "G": "0.25", "B": "0.1"},
        "parameters": {},
        "coordinate_domain": {"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "pixel_width": 64, "pixel_height": 48},
        "rendered_image": rendered,
        "metrics": {"mse": 0.001, "psnr": 30.0, "ssim": 0.95},
    }
    bundle = tmp_path / "bundle"
    export_bundle(model, bundle)
    result = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=bundle,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Verification PASSED" in result.stdout


def test_cli_exports_real_expressions_and_patch_bounds(tmp_path: Path) -> None:
    image = tmp_path / "input.png"
    Image.fromarray((build_sample(96, 64) * 255.0).astype(np.uint8)).save(image)
    out = tmp_path / "out"
    assert cli_main(["-i", str(image), "-o", str(out), "-m", "fidelity", "--max-iter", "3"]) == 0
    project = __import__("json").loads((out / "mathfield_out.mathfield").read_text(encoding="utf-8"))
    assert "sin(x) + cos(y)" not in str(project["global_model"])
    assert "sin(x)" not in str(project["global_model"]["channels"])
    for patch in project["local_patches"]:
        assert len(patch["bounds"]) == 4
        assert all(isinstance(value, float) for value in patch["bounds"])


def test_ablation_flags_change_the_pipeline_configuration() -> None:
    baseline = get_ablation_config("none")
    no_quadtree = get_ablation_config("no_quadtree")
    no_color = get_ablation_config("no_color_decoupling")
    no_varpro = get_ablation_config("no_varpro")
    assert no_quadtree["max_depth"] == 0 and baseline["max_depth"] > 0
    assert no_color["decouple_color"] is False and baseline["decouple_color"] is True
    assert no_varpro["enable_varpro"] is False and baseline["enable_varpro"] is True


def test_demo_scripts_have_no_machine_specific_paths() -> None:
    for name in ("run_lotus_reconstruction.py", "run_web_reconstruction.py"):
        text = (Path(__file__).resolve().parents[1] / name).read_text(encoding="utf-8")
        assert ".gemini" not in text
        assert "C:/Users" not in text
