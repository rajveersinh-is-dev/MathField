"""Regression tests for exporters, CLI honesty and portability."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmarks.ablations import get_ablation_config
from cli import main as cli_main
from examples.generate_sample import build_sample
from mathfield.export.bundle import export_bundle
from mathfield.export.project_file import ProjectFileError, load_project, save_project
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


def test_raster_prefers_channel_equations_over_formula_metadata() -> None:
    model = {"expressions": {"formula": "1", "R": "0.1", "G": "0.5", "B": "0.9"}}
    pixels = np.asarray(_model_to_image(model, width=2, height=2))
    np.testing.assert_array_equal(pixels[0, 0], [25, 127, 229])


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
    assert "Determinism sanity check PASSED" in result.stdout


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


def test_cli_mode_a_bundle_reproduces_fitted_rgb_channels(tmp_path: Path) -> None:
    height, width = 12, 16
    yy, xx = np.mgrid[0:height, 0:width] / np.array([height - 1, width - 1])[:, None, None]
    rgb = np.stack([0.1 + 0.7 * xx, 0.8 - 0.5 * yy, 0.2 + 0.6 * xx * yy], axis=-1)
    image = tmp_path / "mode_a_rgb.png"
    Image.fromarray((rgb * 255.0).astype(np.uint8)).save(image)
    out = tmp_path / "mode_a_rgb"

    assert cli_main(["-i", str(image), "-o", str(out), "-m", "discovery", "--max-iter", "1", "--export-png", "--export-bundle", "-q"]) == 0

    project = json.loads((out / "mathfield_out.mathfield").read_text(encoding="utf-8"))
    channels = project["global_model"]["channels"]
    assert [channel["channel"] for channel in channels] == ["R", "G", "B"]
    assert len({channel["formula"] for channel in channels}) == 3
    assert all(channel["parameters"] for channel in channels)

    namespace = runpy.run_path(str(out / "bundle" / "reproduce.py"))
    reproduced = np.asarray(namespace["reconstruct"](width, height))
    delivered = np.asarray(Image.open(out / "reconstruction.png"))
    np.testing.assert_array_equal(reproduced, delivered)


def test_cli_mode_a_bundle_preserves_rgba_channels(tmp_path: Path) -> None:
    height, width = 10, 14
    yy, xx = np.mgrid[0:height, 0:width] / np.array([height - 1, width - 1])[:, None, None]
    rgba = np.stack([0.1 + 0.7 * xx, 0.8 - 0.5 * yy, 0.2 + 0.6 * xx * yy, 0.3 + 0.4 * xx], axis=-1)
    image = tmp_path / "mode_a_rgba.png"
    Image.fromarray((rgba * 255.0).astype(np.uint8)).save(image)
    out = tmp_path / "mode_a_rgba"

    assert cli_main(["-i", str(image), "-o", str(out), "-m", "discovery", "--max-iter", "1", "--export-png", "--export-bundle", "-q"]) == 0

    project = json.loads((out / "mathfield_out.mathfield").read_text(encoding="utf-8"))
    channels = project["global_model"]["channels"]
    assert [channel["channel"] for channel in channels] == ["R", "G", "B", "A"]
    assert channels[-1]["formula"] != "1"

    namespace = runpy.run_path(str(out / "bundle" / "reproduce.py"))
    reproduced = np.asarray(namespace["reconstruct"](width, height))
    delivered = np.asarray(Image.open(out / "reconstruction.png"))
    assert reproduced.shape == (height, width, 4)
    np.testing.assert_array_equal(reproduced, delivered)


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


@pytest.mark.parametrize(
    ("mode", "flag", "value"),
    [
        ("discovery", "--lambda-reg", "nan"),
        ("discovery", "--lambda-reg", "inf"),
        ("discovery", "--lambda-reg", "-1"),
        ("fidelity", "--lambda-reg", "nan"),
        ("discovery", "--tolerance", "nan"),
        ("fidelity", "--tolerance", "-1"),
        ("discovery", "--max-iter", "0"),
    ],
)
def test_cli_rejects_invalid_numeric_arguments_before_writing_outputs(
    tmp_path: Path,
    mode: str,
    flag: str,
    value: str,
) -> None:
    image = tmp_path / "input.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image)
    output = tmp_path / "out"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "cli.py"), "-i", str(image), "-o", str(output), "-m", mode, flag, value, "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert result.returncode != 0
    assert not output.exists()
    assert not (output / "mathfield_out.mathfield").exists()


def test_project_file_rejects_non_finite_numbers_on_save_and_load(tmp_path: Path) -> None:
    project = {
        "schema_version": "1.0.0",
        "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
        "color_space": "RGB",
        "global_model": {"channels": []},
        "final_metrics": {"mse": float("nan"), "psnr": 100.0},
    }
    target = tmp_path / "invalid.mathfield"
    with pytest.raises(ProjectFileError, match="non-finite"):
        save_project(project, target)
    assert not target.exists()

    target.write_text(json.dumps(project), encoding="utf-8")
    with pytest.raises(ProjectFileError, match="Non-finite"):
        load_project(target)
