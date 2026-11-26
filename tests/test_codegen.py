"""
Unit tests for LaTeX and standalone NumPy code generation.
Verifies parsing by matplotlib.mathtext and isolated sandbox execution.
"""

import numpy as np
import pytest

from mathfield.core.ast.parser import parse
from mathfield.core.codegen.latex import to_latex
from mathfield.core.codegen.numpy_gen import (
    generate_standalone_function,
    generate_standalone_script,
)


def test_latex_formatting():
    # Fractions
    ast1 = parse("(x + 1) / (y + 1)")
    latex1 = to_latex(ast1)
    assert latex1 == r"\frac{x + 1}{y + 1}"

    # Powers
    ast2 = parse("(x + y) ^ 2")
    latex2 = to_latex(ast2)
    assert latex2 == r"\left(x + y\right)^{2}"

    # Parameters & Greek letters
    ast3 = parse("theta0 * sin(x) + c1 * y")
    latex3 = to_latex(ast3)
    assert r"\theta_{0}" in latex3
    assert r"c_{1}" in latex3
    assert r"\sin\left(x\right)" in latex3


def test_matplotlib_mathtext_compatibility():
    try:
        from matplotlib.mathtext import MathTextParser

        parser = MathTextParser("path")
    except ImportError:
        pytest.skip("matplotlib not installed or available.")

    expressions = [
        "c0 * x^2 + c1 * y^2",
        "sin(theta0 * x) * cos(theta1 * y)",
        "rbf_gaussian(x^2 + y^2, 1.5)",
        "rbf_wendland(x, 1.0)",
        "exp(-x) / (1 + exp(-y))",
        "sigmoid(x) + softplus(y)",
        "sqrt(x^2 + 1) + cbrt(y)",
        "clamp(x, -1, 1)",
    ]

    for expr in expressions:
        ast = parse(expr)
        latex_str = to_latex(ast)
        # Verify mathtext can parse and compile the equation
        try:
            parser.parse(f"${latex_str}$")
        except Exception as e:
            pytest.fail(f"matplotlib.mathtext failed to parse LaTeX {latex_str!r} for {expr!r}: {e}")


def test_standalone_numpy_code_sandbox_execution():
    ast = parse("c0 * sin(theta0 * x) + exp(-c1 * (x^2 + y^2)) + 1 / (y + 2.0)")
    params = np.array([2.5, 1.8, 0.7])
    X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))

    # Direct AST evaluation
    expected_z = ast.compile_numpy()(X, Y, params)

    # Generated Python function code
    code_str = generate_standalone_function(ast, func_name="test_eval")
    assert "def test_eval(" in code_str

    # Execute in clean sandbox dictionary with only numpy
    sandbox = {"np": np}
    exec(code_str, sandbox)
    test_eval_fn = sandbox["test_eval"]

    actual_z = test_eval_fn(X, Y, params)
    assert np.allclose(actual_z, expected_z, atol=1e-7)


def test_standalone_reproduction_script_generation():
    ast = parse("c0 * x + c1 * y")
    script = generate_standalone_script(ast, [1.0, 2.0], width=128, height=128)
    assert "import numpy as np" in script
    assert "from PIL import Image" in script
    assert "evaluate_field" in script
    assert "def main():" in script


def test_export_artwork_poster_square_and_portrait(tmp_path):
    from mathfield.export.artwork_poster import export_artwork_poster

    dummy_arr = np.zeros((32, 48, 3), dtype=np.uint8)
    dummy_arr[10:22, 12:36] = 200  # Non-trivial test pattern
    result = {
        "coordinate_domain": {"pixel_width": 48, "pixel_height": 32},
        "rendered_image": dummy_arr,
        "global_equations": {"R": "1.0", "G": "0.5", "B": "0.2"},
        "patch_equations": [{"id": "P_001", "bounds": [-1, -1, 1, 1], "formula": "cos(u)"}],
    }

    out_sq = tmp_path / "poster_sq.png"
    out_pt = tmp_path / "poster_pt.jpg"

    export_artwork_poster(result, out_sq, layout="square", width=400)
    assert out_sq.exists()
    assert out_sq.stat().st_size > 1000

    export_artwork_poster(result, out_pt, layout="portrait", width=300, quality=90)
    assert out_pt.exists()
    assert out_pt.stat().st_size > 1000


def test_artwork_poster_dynamic_formula_panel():
    from PIL import Image

    from mathfield.export.artwork_poster import generate_math_panel

    # 1. With explicit dynamic global and patch equations
    result_with_eqs = {
        "coordinate_domain": {"width": 128, "height": 96},
        "global_equations": {"R": "0.8 T_1(x)", "G": "0.5 T_2(y)", "B": "0.1"},
        "patch_equations": [
            {"id": "L_1", "formula": "0.45 cos(pi*u)"},
            {"id": "L_2", "formula": "0.22 sin(pi*v)"},
        ],
    }
    panel_img = generate_math_panel(result_with_eqs, width_px=800, height_px=400)
    assert isinstance(panel_img, Image.Image)
    assert panel_img.size == (800, 400)

    # 2. Without equations (graceful generic symbolic fallback)
    result_empty = {
        "coordinate_domain": {"width": 64, "height": 64},
    }
    panel_empty = generate_math_panel(result_empty, width_px=800, height_px=400)
    assert isinstance(panel_empty, Image.Image)
    assert panel_empty.size == (800, 400)


def _pdf_report_text(path) -> str:
    try:
        import fitz
    except ImportError:
        pypdf = pytest.importorskip("pypdf")
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    with fitz.open(str(path)) as document:
        return "\n".join(page.get_text() for page in document)


def _mode_a_report_result():
    return {
        "coordinate_domain": {
            "x_min": -1.0,
            "x_max": 1.0,
            "y_min": -1.0,
            "y_max": 1.0,
            "pixel_width": 32,
            "pixel_height": 24,
        },
        "color_space": "RGB",
        "mode": "MODE_A_DISCOVERY",
        "global_equations": {
            "R": "c0 * x + 0.25",
            "G": "0.5",
            "B": "c1 * y - 0.1",
            "formula": "quadtree DCT2Basis metadata",
        },
        "parameters": {
            "R_c0": {"stage": "symbolic", "channel": "R", "value": 0.5},
            "B_c1": {"stage": "symbolic", "channel": "B", "value": -0.2},
        },
        "metrics": {"psnr": 32.5, "ssim": 0.95, "mse": 0.0000295, "mae": 0.001, "linf": 0.004},
        # Stale/auxiliary patch data must not change a Mode A report.
        "patch_equations": [{"id": "P_001", "formula": "DCT2Basis"}],
        "local_patches": [
            {
                "patch_id": 0,
                "bounds": [-1.0, -1.0, 1.0, 1.0],
                "basis": {"type": "dct2", "freq_x": 2, "freq_y": 2},
            }
        ],
        "mode_b_model": {"partition_of_unity": {"separable": False}},
        "local_decomposition": {"partition_of_unity": {"separable": True}},
        "partition_of_unity": {"separable": False},
    }


def _normalized_report_text(text: str) -> str:
    return " ".join(text.casefold().replace(r"\_", "_").split())


def _assert_mode_a_report_text(text: str) -> None:
    normalized = _normalized_report_text(text)
    assert "mode_a_discovery" in normalized
    assert "c0 * x + 0.25" in normalized
    assert "32.50 db" in normalized
    for forbidden in (
        "patch",
        "dct",
        "quadtree",
        "wendland",
        "partition",
        "radial",
        "elliptical",
        "sqrt",
        "fourier",
        "chebyshev",
        "hybrid",
    ):
        assert forbidden not in normalized


def test_mode_a_latex_report_contains_only_delivered_symbolic_model(tmp_path) -> None:
    from mathfield.export.latex_report import export_latex_report

    output = tmp_path / "mode_a.tex"
    export_latex_report(_mode_a_report_result(), output)

    text = output.read_text(encoding="utf-8")
    _assert_mode_a_report_text(text)
    assert "0.500000" in text
    assert "No model parameters were supplied" not in text


def test_mode_a_pdf_report_contains_only_delivered_symbolic_model(tmp_path) -> None:
    from mathfield.export.pdf_report import export_pdf_report

    output = tmp_path / "mode_a.pdf"
    export_pdf_report(_mode_a_report_result(), output)

    _assert_mode_a_report_text(_pdf_report_text(output))


def _serialized_dct_patch():
    return {
        "patch_id": 0,
        "bounds": [-0.5, -0.25, 0.5, 0.25],
        "basis": {
            "type": "dct2",
            "freq_x": 2,
            "freq_y": 2,
            "x_bounds": [-0.5, 0.5],
            "y_bounds": [-0.25, 0.25],
        },
        "coefficients": [[0.1, 0.2, 0.3] for _ in range(9)],
        "coefficient_dimensions": 2,
        "error": 0.0001,
    }


@pytest.mark.parametrize("patch_source", ["mode_b_model", "local_patches"])
def test_mode_b_reports_catalog_only_authoritative_local_patches(tmp_path, patch_source: str) -> None:
    from mathfield.export.latex_report import export_latex_report
    from mathfield.export.pdf_report import export_pdf_report

    patch = _serialized_dct_patch()
    result = {
        "mode": "MODE_B_FIDELITY",
        "global_equations": {"R": "0.5 + 0.1*x", "G": "0.4", "B": "0.3"},
        "parameters": {"R_c0": 0.5},
        "metrics": {"psnr": 30.0},
        # Display-only patch metadata is not an authoritative patch catalog.
        "patch_equations": [{"id": "P_001", "formula": "FABRICATED_DCT2"}],
    }
    if patch_source == "mode_b_model":
        result["mode_b_model"] = {"local_patches": [patch]}
        result["local_patches"] = [{"id": "P_999", "formula": "CONFLICTING_LOCAL_PATCH"}]
    else:
        result["local_patches"] = [patch]

    tex_path = tmp_path / f"mode_b_{patch_source}.tex"
    pdf_path = tmp_path / f"mode_b_{patch_source}.pdf"
    export_latex_report(result, tex_path)
    export_pdf_report(result, pdf_path)

    for text in (tex_path.read_text(encoding="utf-8"), _pdf_report_text(pdf_path)):
        normalized = _normalized_report_text(text)
        assert "p_000" in normalized
        assert "dct2basis" in normalized
        assert "freq_x=2" in normalized
        assert "wendland" in normalized
        assert "partition of unity" in normalized
        assert "p_001" not in normalized
        assert "p_999" not in normalized
        assert "fabricated_dct2" not in normalized
        assert "conflicting_local_patch" not in normalized


@pytest.mark.parametrize("separable", [True, False])
@pytest.mark.parametrize("partition_source", ["mode_b_model", "local_decomposition", "result"])
def test_mode_b_reports_use_persisted_partition_separability(tmp_path, separable: bool, partition_source: str) -> None:
    from mathfield.export.latex_report import export_latex_report
    from mathfield.export.pdf_report import export_pdf_report

    patch = _serialized_dct_patch()
    result = {
        "mode": "MODE_B_FIDELITY",
        "global_equations": {"R": "0.5 + 0.1*x", "G": "0.4", "B": "0.3"},
        "parameters": {"R_c0": 0.5},
        "metrics": {"psnr": 30.0},
        "local_patches": [patch],
    }
    partition = {"overlap_ratio": 0.33, "separable": separable}
    conflicting_partition = {"overlap_ratio": 0.33, "separable": not separable}
    if partition_source == "mode_b_model":
        result["mode_b_model"] = {"local_patches": [patch], "partition_of_unity": partition}
        result["local_decomposition"] = {"partition_of_unity": conflicting_partition}
        result["partition_of_unity"] = conflicting_partition
    elif partition_source == "local_decomposition":
        result["mode_b_model"] = {"local_patches": [patch]}
        result["local_decomposition"] = {"partition_of_unity": partition}
        result["partition_of_unity"] = conflicting_partition
    else:
        result["mode_b_model"] = {"local_patches": [patch]}
        result["partition_of_unity"] = partition

    tex_path = tmp_path / f"mode_b_{partition_source}_{separable}.tex"
    pdf_path = tmp_path / f"mode_b_{partition_source}_{separable}.pdf"
    export_latex_report(result, tex_path)
    export_pdf_report(result, pdf_path)

    reports = (("latex", tex_path.read_text(encoding="utf-8")), ("pdf", _pdf_report_text(pdf_path)))
    for report_format, text in reports:
        normalized = _normalized_report_text(text)
        assert "wendland" in normalized
        assert "partition of unity" in normalized
        if separable:
            assert "separable bivariate" in normalized
            if report_format == "latex":
                assert r"\frac{|x - c_{x,k}|}{r_{x,k}}" in normalized
                assert r"\frac{|y - c_{y,k}|}{r_{y,k}}" in normalized
            else:
                assert "|x - c_{x,k}| / r_{x,k}" in normalized
                assert "|y - c_{y,k}| / r_{y,k}" in normalized
            assert r"\sqrt" not in normalized
            assert "sqrt(" not in normalized
        else:
            assert "radial bivariate" in normalized
            radial_formula = r"\sqrt{dx^2 + dy^2}" if report_format == "latex" else "sqrt(dx^2 + dy^2)"
            assert radial_formula in normalized
            assert "dx = (x - c_{x,k}) / r_{x,k}" in normalized
            assert "dy = (y - c_{y,k}) / r_{y,k}" in normalized
            assert "|x - c_{x,k}|" not in normalized
            assert "|y - c_{y,k}|" not in normalized


def test_mode_b_reports_explicitly_when_local_patches_are_empty(tmp_path) -> None:
    from mathfield.export.latex_report import export_latex_report
    from mathfield.export.pdf_report import export_pdf_report

    result = {
        "mode": "MODE_B_FIDELITY",
        "global_equations": {"R": "0.5 + 0.1*x", "G": "0.4", "B": "0.3"},
        "parameters": {"R_c0": 0.5},
        "metrics": {"psnr": 30.0},
        "mode_b_model": {"local_patches": []},
        "local_patches": [],
        "patch_equations": [{"id": "P_001", "formula": "FABRICATED_DCT2"}],
    }

    tex_path = tmp_path / "mode_b_empty.tex"
    pdf_path = tmp_path / "mode_b_empty.pdf"
    export_latex_report(result, tex_path)
    export_pdf_report(result, pdf_path)

    for text in (tex_path.read_text(encoding="utf-8"), _pdf_report_text(pdf_path)):
        normalized = _normalized_report_text(text)
        assert "no local patches are present" in normalized
        assert "p_001" not in normalized
        assert "dct" not in normalized
        assert "fabricated_dct2" not in normalized
