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
    to_python_code,
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
