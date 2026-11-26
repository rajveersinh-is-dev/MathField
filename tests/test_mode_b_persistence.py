"""Focused persistence and reproduction regressions for complete Mode B models."""

from __future__ import annotations

import ast
import copy
import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import cli
from mathfield.core.ast.parser import parse
from mathfield.core.basis.fourier import DCT2Basis, Fourier2D
from mathfield.core.basis.polynomial import Chebyshev2D
from mathfield.core.checkpoint.serializer import CheckpointSerializer
from mathfield.core.domain import CoordinateDomain2D
from mathfield.core.engine.cpu_numpy import evaluate_numpy
from mathfield.core.local.local_patch import LocalPatch
from mathfield.core.local.partition_of_unity import WendlandPartitionOfUnity
from mathfield.core.model_codec import (
    ModeBCodecError,
    deserialize_mode_b_model,
    serialize_mode_b_model,
)
from mathfield.core.pipeline.mode_b import ModeBPipeline
from mathfield.core.pipeline.stage import PipelineContext, StageStatus
from mathfield.export.project_file import ProjectFileError, load_project, save_project
from mathfield.export.python_script import export_standalone_script


def _render_global_local(context: PipelineContext) -> np.ndarray:
    model = context.global_model
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    matrix = coefficients.reshape(-1, 1) if coefficients.ndim == 1 else coefficients
    x_grid, y_grid = context.domain.to_discrete_grid()
    phi_global = np.hstack(
        [
            model["chebyshev_basis"].basis_matrix(x_grid, y_grid),
            model["fourier_basis"].basis_matrix(x_grid, y_grid)[:, 1:],
        ]
    )
    channel_count = matrix.shape[1]
    global_values = np.empty((x_grid.size, channel_count), dtype=np.float64)
    for channel in range(channel_count):
        global_values[:, channel] = phi_global @ matrix[:, channel]
    composite = global_values.reshape(x_grid.shape + (channel_count,))
    correction = np.zeros_like(composite)

    decomposition = context.local_decomposition or {}
    patches = decomposition.get("patches") or []
    partition = decomposition.get("partition_of_unity") or {
        "overlap_ratio": float(context.config.get("overlap_ratio", 0.20)),
        "separable": bool(context.config.get("separable", True)),
    }
    weights = WendlandPartitionOfUnity(
        overlap_ratio=float(partition["overlap_ratio"]),
        separable=bool(partition["separable"]),
    ).compute_weights(patches, x_grid, y_grid)
    for patch_index, patch in enumerate(patches):
        active = weights[patch_index] > 1e-12
        if not np.any(active):
            continue
        rows, columns = np.where(active)
        row_start, row_stop = int(rows.min()), int(rows.max()) + 1
        column_start, column_stop = int(columns.min()), int(columns.max()) + 1
        sub_x = x_grid[row_start:row_stop, column_start:column_stop]
        sub_y = y_grid[row_start:row_stop, column_start:column_stop]
        sub_fit = patch.evaluate(sub_x, sub_y)
        if sub_fit.ndim == 2:
            sub_fit = sub_fit[:, :, None]
        correction[row_start:row_stop, column_start:column_stop] += (
            weights[patch_index, row_start:row_stop, column_start:column_stop, None] * sub_fit
        )
    return np.clip(composite + correction, 0.0, 1.0)


def _mode_b_context() -> PipelineContext:
    domain = CoordinateDomain2D(width=16, height=4, mode="aspect_preserved")
    chebyshev = Chebyshev2D(deg_x=1, deg_y=1, truncation="total_degree")
    fourier = Fourier2D(freq_x=1, freq_y=1, truncation="rectangular")
    term_count = chebyshev.num_terms + fourier.num_terms - 1
    coefficients = np.zeros((term_count, 4), dtype=np.float64)
    coefficients[0] = [0.20, 0.35, 0.45, 0.60]
    # This wide-domain term is deliberately below the former 1e-4 machine cutoff.
    coefficients[1] = [5e-5, -2e-5, 3e-6, -4e-5]
    for term in range(2, term_count):
        coefficients[term] = np.array([1.0, -0.5, 0.25, 0.125]) * (term * 1e-4)

    patches = [
        LocalPatch(
            bounds=(-4.0, -1.0, 0.8, 1.0),
            basis=DCT2Basis(freq_x=1, freq_y=1),
            coefficients=np.array(
                [
                    [0.08, -0.03, 0.02, 0.01],
                    [-0.04, 0.05, -0.02, 0.03],
                    [0.01, 0.02, 0.03, -0.01],
                    [-0.02, 0.01, 0.04, 0.02],
                ],
                dtype=np.float64,
            ),
            error=0.002,
        ),
        LocalPatch(
            bounds=(-0.8, -1.0, 4.0, 1.0),
            basis=Chebyshev2D(deg_x=1, deg_y=1, truncation="total_degree"),
            coefficients=np.array(
                [
                    [-0.03, 0.04, 0.01, 0.02],
                    [0.02, -0.01, 0.03, -0.02],
                    [0.01, 0.02, -0.01, 0.03],
                ],
                dtype=np.float64,
            ),
            error=0.003,
        ),
    ]
    context = PipelineContext(
        image=np.zeros((4, 16, 4), dtype=np.float64),
        domain=domain,
        config={"overlap_ratio": 0.35, "separable": False},
        global_model={
            "type": "mode_b_global_hybrid",
            "degree": 1,
            "fourier_freq": 1,
            "basis_terms_count": term_count,
            "condition_number": 1.0,
            "coefficients": coefficients,
            "chebyshev_basis": chebyshev,
            "fourier_basis": fourier,
        },
        local_decomposition={"patches": patches, "leaves_count": len(patches)},
        metrics={"mse": 0.001, "psnr": 30.0, "ssim": 0.95},
        condition_report={"condition_number": 1.0},
    )
    context.rendered_image = _render_global_local(context)
    return context


def test_mode_b_project_checkpoint_bundle_and_standalone_roundtrip(tmp_path: Path, monkeypatch) -> None:
    context = _mode_b_context()
    expected = context.rendered_image
    height, width = expected.shape[:2]

    class FakeController:
        def start(self, **kwargs) -> PipelineContext:
            return context

    monkeypatch.setattr(cli, "ReconstructionController", FakeController)
    input_path = tmp_path / "rgba.png"
    Image.fromarray((expected * 255.0).astype(np.uint8), mode="RGBA").save(input_path)
    output = tmp_path / "out"
    assert cli.main(
        [
            "-i",
            str(input_path),
            "-o",
            str(output),
            "-m",
            "fidelity",
            "--max-iter",
            "1",
            "--export-csv",
            "--export-png",
            "--export-bundle",
            "-q",
        ]
    ) == 0

    project = json.loads((output / "mathfield_out.mathfield").read_text(encoding="utf-8"))
    payload = project["mode_b_model"]
    json.dumps(payload, allow_nan=False)
    assert payload["channel_labels"] == ["R", "G", "B", "A"]
    assert payload["partition_of_unity"] == {"overlap_ratio": 0.35, "separable": False}
    np.testing.assert_array_equal(payload["coefficients"], context.global_model["coefficients"])
    assert len(payload["local_patches"]) == 2
    for serialized, original in zip(payload["local_patches"], context.local_decomposition["patches"], strict=True):
        assert serialized["bounds"] == list(original.bounds)
        assert serialized["error"] == original.error
        np.testing.assert_array_equal(serialized["coefficients"], original.coefficients)
    assert [patch["basis"]["type"] for patch in payload["local_patches"]] == ["dct2", "chebyshev2d"]
    assert "rendered_image" not in payload

    channels = project["global_model"]["channels"]
    assert [channel["channel"] for channel in channels] == ["R", "G", "B", "A"]
    for channel_index, channel in enumerate(channels):
        np.testing.assert_array_equal(
            channel["parameters"],
            context.global_model["coefficients"][:, channel_index],
        )

    x_grid, y_grid = context.domain.to_discrete_grid()
    phi_global = np.hstack(
        [
            context.global_model["chebyshev_basis"].basis_matrix(x_grid, y_grid),
            context.global_model["fourier_basis"].basis_matrix(x_grid, y_grid)[:, 1:],
        ]
    )
    for channel_index in range(4):
        expression_value = evaluate_numpy(parse(channels[channel_index]["formula"]), x_grid, y_grid, np.empty(0))
        expected_value = (phi_global @ context.global_model["coefficients"][:, channel_index]).reshape(x_grid.shape)
        np.testing.assert_allclose(expression_value, expected_value, rtol=0.0, atol=1e-10)

    expected_parameter_count = (11 + 4 + 3) * 4
    assert len(project["parameters"]) == expected_parameter_count
    with (output / "parameters.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_parameter_count
    assert {row["channel"] for row in rows} == set("RGBA")
    assert sum(row["stage"] == "local" for row in rows) == 28
    csv_values = {row["parameter_name"]: float(row["parameter_value"]) for row in rows}
    for name, record in project["parameters"].items():
        assert csv_values[name] == pytest.approx(record["value"], rel=1e-12, abs=0.0)

    clone_path = tmp_path / "clone.mathfield"
    save_project(project, clone_path)
    loaded = load_project(clone_path)
    restored_global, restored_local = deserialize_mode_b_model(loaded["mode_b_model"])
    assert isinstance(restored_global["chebyshev_basis"], Chebyshev2D)
    assert isinstance(restored_global["fourier_basis"], Fourier2D)
    assert isinstance(restored_local["patches"][0].basis, DCT2Basis)
    assert isinstance(restored_local["patches"][1].basis, Chebyshev2D)
    restored_context = PipelineContext(image=context.image, domain=context.domain)
    restored_context.global_model = restored_global
    restored_context.local_decomposition = restored_local
    np.testing.assert_allclose(_render_global_local(restored_context), expected, rtol=0.0, atol=1e-10)

    checkpoint = CheckpointSerializer.serialize_context(context, mode="fidelity")
    checkpoint_wire = json.loads(json.dumps(checkpoint))
    assert "rendered_image" not in checkpoint
    assert "rendered_image" not in checkpoint_wire["mode_b_model"]
    checkpoint_context = PipelineContext(image=context.image, domain=context.domain)
    CheckpointSerializer.deserialize_context(checkpoint_wire, checkpoint_context)
    np.testing.assert_array_equal(
        checkpoint_context.global_model["coefficients"],
        context.global_model["coefficients"],
    )
    assert all(isinstance(patch, LocalPatch) for patch in checkpoint_context.local_decomposition["patches"])
    np.testing.assert_allclose(_render_global_local(checkpoint_context), expected, rtol=0.0, atol=1e-10)

    bundle = output / "bundle"
    bundle_project = json.loads((bundle / "project.mathfield").read_text(encoding="utf-8"))
    assert bundle_project["mode_b_model"] == payload
    assert bundle_project["global_model"] == project["global_model"]
    assert bundle_project["local_decomposition"] == project["local_decomposition"]
    assert bundle_project["parameters"] == project["parameters"]

    reproduce_path = bundle / "reproduce.py"
    source = reproduce_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            assert len(node.elts) <= 32
            assert sum(len(item.elts) for item in node.elts if isinstance(item, (ast.List, ast.Tuple))) <= 32
        elif isinstance(node, ast.Dict):
            assert len(node.keys) <= 32
        elif isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "mathfield" for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "mathfield"

    namespace = runpy.run_path(str(reproduce_path))
    reproduced = np.asarray(namespace["reconstruct"](width, height))
    delivered = np.asarray(Image.open(output / "reconstruction.png"))
    assert reproduced.shape == (height, width, 4)
    assert int(np.max(np.abs(reproduced.astype(np.int16) - delivered.astype(np.int16)))) <= 1

    verified = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=bundle,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr
    assert "Rendered pixel parity PASSED" in verified.stdout
    assert "global analytical component" not in verified.stdout

    tampered = source.replace(
        "global_coefficients[0, 0] = 0.20000000000000001",
        "global_coefficients[0, 0] = 0.25000000000000000",
        1,
    )
    assert tampered != source
    # Change the file size as well so same-second Python bytecode caches cannot
    # reuse the pre-tamper module during this negative verification check.
    reproduce_path.write_text(tampered + "\n# tampered reference check\n", encoding="utf-8")
    rejected = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=bundle,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "differ from the delivered reference" in rejected.stderr


@pytest.mark.parametrize("separable", [True, False])
def test_actual_mode_b_pipeline_standalone_matches_rendered_pixels(tmp_path: Path, separable: bool) -> None:
    height, width = 10, 12
    y_grid, x_grid = np.mgrid[0:height, 0:width] / np.array([height - 1, width - 1])[:, None, None]
    rgba = np.stack(
        [
            0.15 + 0.65 * x_grid,
            0.75 - 0.45 * y_grid,
            0.25 + 0.35 * x_grid * y_grid,
            0.35 + 0.45 * x_grid,
        ],
        axis=-1,
    )
    domain = CoordinateDomain2D(width=width, height=height)
    config = {
        "global_degree": 1,
        "fourier_freq": 1,
        "max_depth": 1,
        "split_threshold": 0.0,
        "local_freq": 1,
        "overlap_ratio": 0.20,
        "separable": separable,
    }
    context = PipelineContext(image=rgba, domain=domain, config=config)
    assert ModeBPipeline(config=config).execute(context) == StageStatus.COMPLETED
    assert len(context.local_decomposition["patches"]) >= 4
    assert context.local_decomposition["partition_of_unity"]["separable"] is separable

    payload = serialize_mode_b_model(
        context.global_model,
        context.local_decomposition,
        channel_labels=list("RGBA"),
        partition={"overlap_ratio": 0.20, "separable": separable},
    )
    assert payload["partition_of_unity"]["separable"] is separable
    restored_global, restored_local = deserialize_mode_b_model(payload)
    restored = PipelineContext(image=rgba, domain=domain)
    restored.global_model = restored_global
    restored.local_decomposition = restored_local
    np.testing.assert_allclose(_render_global_local(restored), context.rendered_image, rtol=0.0, atol=1e-10)

    checkpoint_wire = json.loads(json.dumps(CheckpointSerializer.serialize_context(context, mode="fidelity")))
    checkpoint_context = PipelineContext(image=rgba, domain=domain, config=config)
    CheckpointSerializer.deserialize_context(checkpoint_wire, checkpoint_context)
    assert checkpoint_context.local_decomposition["partition_of_unity"]["separable"] is separable
    np.testing.assert_allclose(_render_global_local(checkpoint_context), context.rendered_image, rtol=0.0, atol=1e-10)

    script_path = tmp_path / "reproduce.py"
    export_standalone_script(
        {
            "mode_b_model": payload,
            "coordinate_domain": {
                **domain.to_dict(),
                "pixel_width": width,
                "pixel_height": height,
            },
        },
        script_path,
    )
    namespace = runpy.run_path(str(script_path))
    reproduced = np.asarray(namespace["reconstruct"](width, height))
    delivered = np.clip(context.rendered_image * 255.0, 0.0, 255.0).astype(np.uint8)
    assert reproduced.shape == delivered.shape == (height, width, 4)
    assert int(np.max(np.abs(reproduced.astype(np.int16) - delivered.astype(np.int16)))) <= 1


def test_mode_b_standalone_materializes_global_bases_in_bounded_blocks(tmp_path: Path) -> None:
    chebyshev = Chebyshev2D(deg_x=6, deg_y=6, truncation="total_degree")
    fourier = Fourier2D(freq_x=12, freq_y=12, truncation="rectangular")
    term_count = chebyshev.num_terms + fourier.num_terms - 1
    payload = serialize_mode_b_model(
        {
            "type": "mode_b_global_hybrid",
            "degree": 6,
            "fourier_freq": 12,
            "basis_terms_count": term_count,
            "condition_number": 1.0,
            "coefficients": np.zeros((term_count, 3)),
            "chebyshev_basis": chebyshev,
            "fourier_basis": fourier,
        },
        channel_labels=list("RGB"),
    )
    script_path = tmp_path / "reproduce.py"
    export_standalone_script(
        {
            "mode_b_model": payload,
            "coordinate_domain": {
                "x_min": -1.0,
                "x_max": 1.0,
                "y_min": -1.0,
                "y_max": 1.0,
                "pixel_width": 256,
                "pixel_height": 192,
            },
        },
        script_path,
    )
    namespace = runpy.run_path(str(script_path))
    observed_rows: list[int] = []
    original_chebyshev = namespace["_chebyshev_matrix"]
    original_fourier = namespace["_fourier_matrix"]

    def tracked_chebyshev(*args):
        observed_rows.append(args[0].size)
        return original_chebyshev(*args)

    def tracked_fourier(*args):
        observed_rows.append(args[0].size)
        return original_fourier(*args)

    function_globals = namespace["reconstruct"].__globals__
    function_globals["_chebyshev_matrix"] = tracked_chebyshev
    function_globals["_fourier_matrix"] = tracked_fourier
    image = namespace["reconstruct"](256, 192)
    assert np.asarray(image).shape == (192, 256, 3)
    assert observed_rows
    assert max(observed_rows) <= 4096


def test_mode_b_codec_validates_basis_coefficient_and_partition_shapes() -> None:
    payload = serialize_mode_b_model(
        _mode_b_context().global_model,
        _mode_b_context().local_decomposition,
        channel_labels=list("RGBA"),
        partition={"overlap_ratio": 0.35, "separable": False},
    )

    mutations = [
        lambda data: data.update(codec_version="99.0.0"),
        lambda data: data["global_basis"]["chebyshev"].update(deg_x=-1),
        lambda data: data["global_basis"]["fourier"].update(truncation="diagonal"),
        lambda data: data["global_basis"].update(include_fourier_dc="false"),
        lambda data: data["coefficients"].pop(),
        lambda data: data["local_patches"][0]["bounds"].__setitem__(2, -4.0),
        lambda data: data["local_patches"][0]["basis"].update(freq_x=-1),
        lambda data: data["partition_of_unity"].update(overlap_ratio=-0.1),
    ]
    for mutate in mutations:
        invalid = copy.deepcopy(payload)
        mutate(invalid)
        with pytest.raises(ModeBCodecError):
            deserialize_mode_b_model(invalid)

    shared_context = _mode_b_context()
    shared_context.local_decomposition["patches"][0].coefficients = np.array([0.01, -0.02, 0.03, 0.04])
    shared_payload = serialize_mode_b_model(
        shared_context.global_model,
        shared_context.local_decomposition,
        channel_labels=list("RGBA"),
    )
    shared_coefficients = np.asarray(shared_payload["local_patches"][0]["coefficients"])
    assert shared_coefficients.shape == (4, 4)
    for channel in range(4):
        np.testing.assert_array_equal(shared_coefficients[:, channel], [0.01, -0.02, 0.03, 0.04])

    invalid_project = {
        "schema_version": "1.0.0",
        "coordinate_domain": {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0},
        "color_space": "RGBA",
        "global_model": {"channels": []},
        "final_metrics": {"mse": 0.0, "psnr": 100.0},
        "mode_b_model": {**payload, "codec_version": "99.0.0"},
    }
    with pytest.raises(ProjectFileError, match="Mode B model payload"):
        save_project(invalid_project, Path("unused.mathfield"))
