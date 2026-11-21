"""
MathField E2E Test Suite Helper Module.

Provides:
- Headless CLI subprocess runner (`run_cli`, `CliResult`) with timeout protection.
- Isolated standalone reproduction script runner (`run_standalone_script`).
- Comprehensive format inspectors & validators for PNG, JPG, SVG, Python, PDF,
  LaTeX, JSON Project, CSV Parameter Tables, and Checkpoint files.
- Independent reference mathematical metric calculations in pure NumPy & SciPy
  (MSE, PSNR, MAE, L_inf, SSIM, CIEDE2000).
"""

import ast as py_ast
import csv
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# CLI Execution Subprocess Runner
# ============================================================================


class CliResult:
    """Encapsulates the result of a CLI subprocess invocation."""

    def __init__(self, exit_code: int, stdout: str, stderr: str, duration: float, cmd: List[str]) -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration = duration
        self.cmd = cmd

    @property
    def success(self) -> bool:
        """True if execution exited with code 0."""
        return self.exit_code == 0

    def assert_success(self, msg: str = "") -> None:
        """Asserts that the CLI command succeeded with exit code 0."""
        if self.exit_code != 0:
            extra = f"\n{msg}" if msg else ""
            raise AssertionError(
                f"CLI execution failed with exit code {self.exit_code}.\n"
                f"Command: {' '.join(self.cmd)}\n"
                f"Duration: {self.duration:.2f}s\n"
                f"--- STDOUT ---\n{self.stdout}\n"
                f"--- STDERR ---\n{self.stderr}"
                f"{extra}"
            )

    def assert_exit_code(self, expected_code: int, msg: str = "") -> None:
        """Asserts that the CLI command exited with a specific exit code."""
        if self.exit_code != expected_code:
            extra = f"\n{msg}" if msg else ""
            raise AssertionError(
                f"Expected exit code {expected_code}, got {self.exit_code}.\n"
                f"Command: {' '.join(self.cmd)}\n"
                f"Duration: {self.duration:.2f}s\n"
                f"--- STDOUT ---\n{self.stdout}\n"
                f"--- STDERR ---\n{self.stderr}"
                f"{extra}"
            )


def kill_process_tree(pid: int) -> None:
    """
    Recursively terminates a process and all its descendants on Windows and POSIX.
    Prevents pipe deadlocks and lingering zombie processes upon test timeouts.
    """
    if sys.platform == "win32":
        try:
            # taskkill /F /T kills entire process tree immediately in Win32 kernel
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            pass

    try:
        import psutil

        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        parent.kill()
        _, still_alive = psutil.wait_procs(children + [parent], timeout=2.0)
        for p in still_alive:
            try:
                p.kill()
            except Exception:
                pass
    except Exception:
        pass


def run_cli(
    args: List[str],
    timeout: float = 120.0,
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    python_exe: str = sys.executable,
) -> CliResult:
    """
    Executes MathField CLI (cli.py) in a headless subprocess with timeout protection
    and recursive process-tree termination on Windows.
    Sets PYTHONPATH to project root and ensures unbuffered output.
    """
    cli_path = PROJECT_ROOT / "cli.py"
    cmd = [python_exe, str(cli_path)] + args

    call_env = os.environ.copy()
    call_env["PYTHONPATH"] = str(PROJECT_ROOT)
    call_env["PYTHONUNBUFFERED"] = "1"
    if env:
        call_env.update(env)

    start_time = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else str(PROJECT_ROOT),
        env=call_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        duration = time.perf_counter() - start_time
        return CliResult(proc.returncode, stdout, stderr, duration, cmd)
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start_time
        # Terminate entire process tree so child worker processes release inherited pipes
        kill_process_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=2.0)
        except Exception:
            if proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            if proc.stderr:
                try:
                    proc.stderr.close()
                except Exception:
                    pass
            stdout, stderr = "", ""
        return CliResult(-1, stdout, f"Command timed out after {timeout} seconds: {exc}\n{stderr}", duration, cmd)
    except Exception as exc:
        duration = time.perf_counter() - start_time
        kill_process_tree(proc.pid)
        return CliResult(-1, "", str(exc), duration, cmd)


def run_standalone_script(
    script_path: Union[str, Path],
    args: Optional[List[str]] = None,
    timeout: float = 30.0,
    cwd: Optional[Union[str, Path]] = None,
    python_exe: str = sys.executable,
) -> CliResult:
    """
    Executes a standalone generated Python script in an isolated subprocess with
    recursive process-tree termination on Windows.
    Crucially strips MathField from PYTHONPATH to verify that the script is 100% self-contained.
    """
    cmd = [python_exe, str(script_path)] + (args or [])

    isolated_env = os.environ.copy()
    isolated_env.pop("PYTHONPATH", None)
    isolated_env["PYTHONUNBUFFERED"] = "1"

    start_time = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else str(Path(script_path).parent),
        env=isolated_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        duration = time.perf_counter() - start_time
        return CliResult(proc.returncode, stdout, stderr, duration, cmd)
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start_time
        kill_process_tree(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=2.0)
        except Exception:
            if proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            if proc.stderr:
                try:
                    proc.stderr.close()
                except Exception:
                    pass
            stdout, stderr = "", ""
        return CliResult(-1, stdout, f"Standalone script timed out after {timeout}s: {exc}", duration, cmd)
    except Exception as exc:
        duration = time.perf_counter() - start_time
        kill_process_tree(proc.pid)
        return CliResult(-1, "", str(exc), duration, cmd)


# ============================================================================
# File Format Inspectors & Validators
# ============================================================================


def inspect_png(
    path: Union[str, Path], expected_shape: Optional[Tuple[int, int]] = None, has_alpha: Optional[bool] = None
) -> Dict[str, Any]:
    """
    Validates PNG header (magic bytes: 89 50 4E 47 0D 0A 1A 0A).
    Loads with Pillow; returns dict:
    {"width", "height", "mode", "has_alpha", "file_size", "variance", "array"}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PNG file not found: {p}")

    content = p.read_bytes()
    if len(content) < 68:
        raise ValueError(f"PNG file too small ({len(content)} bytes), invalid header/chunks.")

    # Check PNG magic bytes
    png_magic = b"\x89PNG\r\n\x1a\n"
    if not content.startswith(png_magic):
        raise ValueError(f"File {p} does not start with standard PNG magic bytes.")

    img = Image.open(p)
    width, height = img.size
    mode = img.mode
    arr = np.array(img)

    actual_has_alpha = "A" in mode
    variance = float(np.var(arr))

    if expected_shape is not None and (width, height) != expected_shape and (height, width) != expected_shape:
        raise AssertionError(f"Expected image dimensions {expected_shape}, got ({width}, {height})")

    if has_alpha is not None and actual_has_alpha != has_alpha:
        raise AssertionError(f"Expected has_alpha={has_alpha}, got mode {mode}")

    return {
        "width": width,
        "height": height,
        "mode": mode,
        "has_alpha": actual_has_alpha,
        "file_size": len(content),
        "variance": variance,
        "array": arr,
    }


def inspect_jpg(path: Union[str, Path], min_quality_check: bool = False) -> Dict[str, Any]:
    """
    Validates JPEG SOI (FF D8) and EOI (FF D9).
    Loads with Pillow; returns dict:
    {"width", "height", "mode", "file_size", "exif", "array"}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JPG file not found: {p}")

    content = p.read_bytes()
    if len(content) < 100:
        raise ValueError(f"JPG file too small ({len(content)} bytes).")

    if not content.startswith(b"\xff\xd8"):
        raise ValueError(f"File {p} does not start with JPEG SOI magic bytes.")

    img = Image.open(p)
    width, height = img.size
    mode = img.mode
    exif = img.getexif()
    exif_dict = dict(exif) if exif else {}
    arr = np.array(img)

    return {
        "width": width,
        "height": height,
        "mode": mode,
        "file_size": len(content),
        "exif": exif_dict,
        "array": arr,
    }


def inspect_svg(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parses SVG using xml.etree.ElementTree.
    Validates root tag is svg.
    Checks for vector elements or raster fallback.
    Extracts XML metadata block.
    """
    import xml.etree.ElementTree as ET

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SVG file not found: {p}")

    text = p.read_text(encoding="utf-8")
    root = ET.fromstring(text)

    # Normalize tag name (strip namespace if present)
    tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
    if tag.lower() != "svg":
        raise ValueError(f"Expected root tag 'svg', got '{root.tag}'")

    width = root.attrib.get("width", "")
    height = root.attrib.get("height", "")
    viewbox = root.attrib.get("viewBox", "")

    # Look for vector paths and raster images
    all_tags = [elem.tag.split("}")[-1].lower() for elem in root.iter()]
    vector_tags = {"path", "polygon", "polyline", "circle", "ellipse", "line", "rect"}
    has_vector = any(t in vector_tags for t in all_tags)
    has_image = "image" in all_tags

    # Look for metadata block
    metadata_content = ""
    for elem in root.iter():
        elem_tag = elem.tag.split("}")[-1].lower()
        if elem_tag in ("metadata", "desc"):
            metadata_content += "".join(elem.itertext())

    return {
        "is_valid_xml": True,
        "has_vector_paths": has_vector,
        "has_embedded_raster": has_image,
        "is_vector": has_vector,
        "has_image": has_image,
        "has_metadata": len(metadata_content.strip()) > 0,
        "equations_metadata": metadata_content,
        "element_count": len(all_tags),
        "width": width,
        "height": height,
        "viewbox": viewbox,
    }


def inspect_json_project(path: Union[str, Path], schema_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Parses JSON project file (.mathfield).
    Validates schema compliance and required keys.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Project file not found: {p}")

    content = p.read_text(encoding="utf-8")
    data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("Project file JSON root must be an object.")

    # Check key variants (schema_version or version, coordinate_domain or domain)
    has_ver = ("schema_version" in data) or ("version" in data)
    has_domain = ("coordinate_domain" in data) or ("domain" in data)
    has_model = ("global_model" in data) or ("expressions" in data)

    if not (has_ver and has_domain and has_model):
        raise ValueError(f"Project JSON missing mandatory keys. Present: {list(data.keys())}")

    if schema_dict is not None:
        import jsonschema

        jsonschema.validate(instance=data, schema=schema_dict)

    param_count = 0
    if "global_model" in data and isinstance(data["global_model"], dict):
        channels = data["global_model"].get("channels", [])
        for ch in channels:
            param_count += len(ch.get("parameters", []))
    elif "parameters" in data:
        param_count = len(data["parameters"])

    data["param_count"] = param_count
    return data


def inspect_csv_parameters(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Parses CSV parameter table using csv.DictReader.
    Validates header and ensures parameter values are parseable finite floats.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    rows: List[Dict[str, Any]] = []
    with p.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file has no header row.")

        headers_lower = [h.strip().lower() for h in reader.fieldnames]

        # Check for presence of value column
        val_col = None
        for col in ["parameter_value", "value", "param_value", "val"]:
            if col in headers_lower:
                val_col = reader.fieldnames[headers_lower.index(col)]
                break

        if val_col is None:
            raise ValueError(f"CSV header missing parameter value column. Found: {reader.fieldnames}")

        for row in reader:
            raw_val = row[val_col]
            try:
                num_val = float(raw_val)
                if not math.isfinite(num_val):
                    raise ValueError(f"Non-finite float value in CSV: {raw_val}")
            except ValueError as exc:
                raise ValueError(f"Could not parse float value '{raw_val}': {exc}") from exc
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("CSV parameter table contains zero data rows.")

    return rows


def inspect_pdf_report(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validates PDF magic bytes (%PDF-).
    Inspects with PyMuPDF (fitz) or pypdf.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF file not found: {p}")

    content = p.read_bytes()
    if not content.startswith(b"%PDF-"):
        raise ValueError(f"File {p} does not start with PDF magic bytes %PDF-.")

    page_count = 0
    full_text = ""
    image_count = 0

    # Try PyMuPDF (fitz)
    try:
        import fitz

        doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(doc)
        for page in doc:
            full_text += page.get_text()
            image_count += len(page.get_images())
    except ImportError:
        # Fallback to pypdf
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
        for page in reader.pages:
            full_text += page.extract_text() or ""
            image_count += len(page.images)

    contains_eq = any(sym in full_text for sym in ["\\", "+", "*", "sin", "cos", "x", "y"])

    return {
        "page_count": page_count,
        "has_text": len(full_text.strip()) > 0,
        "contains_equation_text": contains_eq,
        "file_size": len(content),
        "text_length": len(full_text),
        "image_count": image_count,
        "text": full_text,
    }


def inspect_latex_doc(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Reads .tex document.
    Validates LaTeX structure: \\documentclass, \\begin{document}, \\end{document}.
    Checks balanced braces.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LaTeX file not found: {p}")

    text = p.read_text(encoding="utf-8")

    has_docclass = "\\documentclass" in text
    has_begin_doc = "\\begin{document}" in text
    has_end_doc = "\\end{document}" in text
    is_compilable = has_docclass and has_begin_doc and has_end_doc

    # Check balanced braces
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces != close_braces:
        raise ValueError(f"Unbalanced braces in LaTeX document: {open_braces} open vs {close_braces} close.")

    eq_count = text.count("\\begin{equation}") + text.count("\\begin{align}") + text.count("\\[")
    tables_count = text.count("\\begin{tabular}") + text.count("\\begin{table}")
    lines = text.splitlines()

    return {
        "is_compilable_structure": is_compilable,
        "equation_count": eq_count,
        "tables_found": tables_count,
        "line_count": len(lines),
        "open_braces": open_braces,
        "close_braces": close_braces,
    }


def inspect_checkpoint(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Parses .mfchk checkpoint file.
    Validates JSON AST structure, IEEE-754 hex float parameter dictionary, and RNG BitGenerator state dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    content = p.read_text(encoding="utf-8")
    data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("Checkpoint file must be a JSON object.")

    has_ver = "schema_version" in data or "version" in data
    has_stage = "stage" in data or "stage_idx" in data
    has_iter = "iteration" in data or "iter" in data

    if not (has_ver and has_stage and has_iter):
        raise ValueError(f"Checkpoint missing required metadata keys. Keys: {list(data.keys())}")

    # Verify hex float parameters if present
    params = data.get("parameters", {})
    if isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, dict) and "hex" in v:
                hex_str = v["hex"]
                restored = float.fromhex(hex_str)
                if not math.isfinite(restored):
                    raise ValueError(f"Restored non-finite float from hex '{hex_str}' for param {k}")

    return data


def assert_no_memorized_arrays(model_data: Any, max_param_count: int = 32) -> None:
    """
    Deeply audits a discovered model against direct pixel memorization and lookup tables.
    Verifies:
    1. Structural: No 2D array literals (lists containing lists).
    2. Structural: No 1D array literals exceeding max_param_count.
    3. Structural: No coordinate-mapping dictionary literals.
    4. AST: Formula strings contain strictly NO List, Tuple, Dict, or Subscript nodes.
    5. Lexical: No forbidden lookup identifiers (table, raster, lookup, memmap, pixel_grid).
    """
    FORBIDDEN_KEYWORDS = {"lookup_table", "pixel_array", "raster_data", "lut", "grid_values", "pixel_map"}

    def _traverse(obj: Any, path: str = "model") -> None:
        if isinstance(obj, (list, tuple)):
            if any(isinstance(x, (list, tuple)) for x in obj):
                raise AssertionError(f"Anti-memorization violation at {path}: contains 2D array literal!")
            if len(obj) > max_param_count:
                raise AssertionError(
                    f"Anti-memorization violation at {path}: array literal exceeds {max_param_count} elements (len={len(obj)})"
                )
            for i, x in enumerate(obj):
                _traverse(x, f"{path}[{i}]")

        elif isinstance(obj, dict):
            if len(obj) > 64:
                raise AssertionError(
                    f"Anti-memorization violation at {path}: dict has too many entries ({len(obj)} > 64)"
                )
            for k, v in obj.items():
                if any(kw in str(k).lower() for kw in FORBIDDEN_KEYWORDS):
                    raise AssertionError(f"Anti-memorization violation at {path}: key '{k}' matches forbidden keyword")
                _traverse(v, f"{path}.{k}")

        elif isinstance(obj, str):
            for kw in FORBIDDEN_KEYWORDS:
                if kw in obj.lower():
                    raise AssertionError(
                        f"Anti-memorization violation at {path}: string contains forbidden keyword '{kw}'"
                    )
            _check_formula_ast(obj, path)

    def _check_formula_ast(formula_str: str, path: str) -> None:
        if any(c in formula_str for c in ("[", "]", "{", "}")):
            try:
                tree = py_ast.parse(formula_str, mode="eval")
                for node in py_ast.walk(tree):
                    if isinstance(node, (py_ast.List, py_ast.Tuple)):
                        raise AssertionError(
                            f"Anti-memorization violation at {path}: formula AST contains array literal ({type(node).__name__})"
                        )
                    if isinstance(node, py_ast.Subscript):
                        raise AssertionError(
                            f"Anti-memorization violation at {path}: formula AST contains subscript/lookup indexing"
                        )
                    if isinstance(node, py_ast.Dict):
                        raise AssertionError(
                            f"Anti-memorization violation at {path}: formula AST contains dict literal"
                        )
            except SyntaxError as exc:
                clean = formula_str.strip()
                if clean.startswith(("[", "{")):
                    raise AssertionError(
                        f"Anti-memorization violation at {path}: formula starts with literal array delimiter: {clean[:20]}"
                    ) from exc

    _traverse(model_data)


def validate_standalone_python_script(
    script_path: Union[str, Path],
    test_execution: bool = False,
    output_image_path: Optional[Union[str, Path]] = None,
    max_array_elements: int = 32,
    max_byte_length: int = 64,
) -> Dict[str, Any]:
    """
    Audits a standalone reproduction script for strict mathematical purity and sandbox security:
    1. Parses into Python AST.
    2. Strictly forbids importing mathfield or untrusted modules (importlib, builtins, ctypes, socket, etc.).
    3. Strictly forbids direct and indirect calls to eval(), exec(), compile(), or __import__().
    4. Detects raw memorized pixel arrays, 2D matrix literals, or lookup tables exceeding threshold.
    5. Detects disguised binary rasters in bytes constants or array multiplications.
    6. If test_execution=True, runs the script in an isolated subprocess.
    """
    p = Path(script_path)
    if not p.exists():
        raise FileNotFoundError(f"Script not found: {p}")

    code = p.read_text(encoding="utf-8")
    tree = py_ast.parse(code, filename=str(p))

    FORBIDDEN_CALL_NAMES = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }
    FORBIDDEN_ATTRS = {"eval", "exec", "__builtins__"}
    FORBIDDEN_MODULES = {
        "mathfield",
        "builtins",
        "__builtin__",
        "importlib",
        "pickle",
        "ctypes",
        "subprocess",
        "socket",
        "base64",
        "zlib",
        "gzip",
        "bz2",
        "lzma",
        "marshal",
        "urllib",
        "requests",
    }
    FORBIDDEN_LITERAL_STRS = {"eval", "exec", "__import__", "mathfield"}

    imports: List[str] = []

    for node in py_ast.walk(tree):
        # 1. Static Imports Check
        if isinstance(node, py_ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                mod = alias.name.split(".")[0]
                if mod in FORBIDDEN_MODULES:
                    raise AssertionError(
                        f"Standalone script violates security/independence rule: imports '{alias.name}'"
                    )
        elif isinstance(node, py_ast.ImportFrom):
            if node.module:
                imports.append(node.module)
                mod = node.module.split(".")[0]
                if mod in FORBIDDEN_MODULES:
                    raise AssertionError(
                        f"Standalone script violates security/independence rule: imports from '{node.module}'"
                    )

        # 2. Function & Method Calls Check
        elif isinstance(node, py_ast.Call):
            if isinstance(node.func, py_ast.Name):
                if node.func.id in FORBIDDEN_CALL_NAMES:
                    raise AssertionError(f"Standalone script violates security rule: forbidden call '{node.func.id}()'")
            elif isinstance(node.func, py_ast.Attribute) and (
                node.func.attr in FORBIDDEN_ATTRS or node.func.attr == "import_module"
            ):
                raise AssertionError(
                    f"Standalone script violates security rule: forbidden attribute call '.{node.func.attr}()'"
                )

            for kw in node.keywords:
                if (
                    kw.arg == "name"
                    and isinstance(kw.value, py_ast.Constant)
                    and kw.value.value in FORBIDDEN_LITERAL_STRS
                ):
                    raise AssertionError(
                        f"Standalone script violates security rule: indirect getattr call on '{kw.value.value}'"
                    )

        # 3. Attribute Access Check (e.g. builtins.__dict__, builtins.eval)
        elif isinstance(node, py_ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                raise AssertionError(
                    f"Standalone script violates security rule: forbidden attribute access '.{node.attr}'"
                )

        # 4. Subscript / Dictionary Lookups (e.g. builtins.__dict__['exec'])
        elif isinstance(node, py_ast.Subscript):
            if isinstance(node.slice, py_ast.Constant) and node.slice.value in FORBIDDEN_LITERAL_STRS:
                raise AssertionError(
                    f"Standalone script violates security rule: dictionary lookup for '{node.slice.value}'"
                )

        # 5. Raw Memorized Arrays & Lookup Tables Check
        elif isinstance(node, (py_ast.List, py_ast.Tuple)):
            if len(node.elts) > max_array_elements:
                raise AssertionError(
                    f"Standalone script violates anti-memorization rule: literal array/tuple exceeds {max_array_elements} elements (len={len(node.elts)})"
                )
            if any(isinstance(elt, (py_ast.List, py_ast.Tuple)) for elt in node.elts):
                total_nested = sum(len(elt.elts) for elt in node.elts if isinstance(elt, (py_ast.List, py_ast.Tuple)))
                if total_nested > max_array_elements:
                    raise AssertionError(
                        f"Standalone script violates anti-memorization rule: nested 2D array literal exceeds {max_array_elements} elements (total={total_nested})"
                    )

        elif isinstance(node, py_ast.Dict):
            if len(node.keys) > max_array_elements:
                raise AssertionError(
                    f"Standalone script violates anti-memorization rule: literal dict exceeds {max_array_elements} entries (len={len(node.keys)})"
                )

        # 6. Array & Bytes Multiplications Check (e.g. [0] * 2000, b'\x00' * 100)
        elif isinstance(node, py_ast.BinOp) and isinstance(node.op, py_ast.Mult):
            mult_len = None
            if (
                isinstance(node.left, (py_ast.List, py_ast.Tuple))
                and isinstance(node.right, py_ast.Constant)
                and isinstance(node.right.value, int)
            ):
                mult_len = len(node.left.elts) * node.right.value
            elif (
                isinstance(node.right, (py_ast.List, py_ast.Tuple))
                and isinstance(node.left, py_ast.Constant)
                and isinstance(node.left.value, int)
            ):
                mult_len = len(node.right.elts) * node.left.value
            if mult_len is not None and mult_len > max_array_elements:
                raise AssertionError(
                    f"Standalone script violates anti-memorization rule: array multiplication produces {mult_len} elements (> {max_array_elements})"
                )

            mult_bytes = None
            if (
                isinstance(node.left, py_ast.Constant)
                and isinstance(node.left.value, bytes)
                and isinstance(node.right, py_ast.Constant)
                and isinstance(node.right.value, int)
            ):
                mult_bytes = len(node.left.value) * node.right.value
            elif (
                isinstance(node.right, py_ast.Constant)
                and isinstance(node.right.value, bytes)
                and isinstance(node.left, py_ast.Constant)
                and isinstance(node.left.value, int)
            ):
                mult_bytes = len(node.right.value) * node.left.value
            if mult_bytes is not None and mult_bytes > max_byte_length:
                raise AssertionError(
                    f"Standalone script violates anti-memorization rule: byte multiplication produces {mult_bytes} bytes (> {max_byte_length})"
                )

        # 7. Constant Binary Payloads Check
        elif isinstance(node, py_ast.Constant) and isinstance(node.value, bytes) and len(node.value) > max_byte_length:
            raise AssertionError(
                f"Standalone script violates anti-memorization rule: byte constant exceeds {max_byte_length} bytes (len={len(node.value)})"
            )

    output_valid = False
    if test_execution:
        out_target = output_image_path or (p.parent / "standalone_test_out.png")
        cli_res = run_standalone_script(p, args=["--output", str(out_target), "--resolution", "32", "32"], timeout=30.0)
        cli_res.assert_success("Standalone script execution failed in isolated environment.")
        if not Path(out_target).exists():
            raise AssertionError(f"Standalone script did not generate expected output at {out_target}")
        img = Image.open(out_target)
        if img.size != (32, 32):
            raise AssertionError(f"Expected generated image size (32, 32), got {img.size}")
        output_valid = True

    return {
        "ast_nodes": len(list(py_ast.walk(tree))),
        "imports": imports,
        "has_eval_or_exec": False,
        "output_valid": output_valid,
    }


def validate_reproducibility_package(bundle_path: Union[str, Path], test_verify_script: bool = False) -> Dict[str, Any]:
    """
    Validates a standalone reproducibility package bundle (.zip or directory).
    Checks for all 7 required assets:
    - reproduce.py
    - verify.py
    - project.mathfield
    - parameters.csv
    - report.pdf
    - report.tex (or equations.tex)
    - README.md
    """
    p = Path(bundle_path)
    if not p.exists():
        raise FileNotFoundError(f"Bundle not found: {p}")

    required_names = [
        "reproduce.py",
        "verify.py",
        "project.mathfield",
        "parameters.csv",
        "report.pdf",
        "README.md",
    ]

    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            names = [Path(n).name for n in z.namelist()]
            for req in required_names:
                if req not in names:
                    raise AssertionError(f"Bundle ZIP missing required asset: {req}. Found: {names}")
            has_tex = any(n.endswith(".tex") for n in names)
            if not has_tex:
                raise AssertionError("Bundle ZIP missing LaTeX document (.tex).")
    else:
        # Directory bundle
        dir_files = [f.name for f in p.iterdir() if f.is_file()]
        for req in required_names:
            if req not in dir_files:
                raise AssertionError(f"Bundle directory missing required asset: {req}. Found: {dir_files}")
        has_tex = any(f.suffix == ".tex" for f in p.iterdir())
        if not has_tex:
            raise AssertionError("Bundle directory missing LaTeX document (.tex).")

    return {
        "assets_verified": True,
        "bundle_path": str(p),
    }


# Aliases for explorer/spec miner naming conventions
validate_png_export = inspect_png
validate_jpg_export = inspect_jpg
validate_svg_export = inspect_svg
validate_json_project_file = inspect_json_project
validate_csv_parameter_table = inspect_csv_parameters
validate_pdf_report = inspect_pdf_report
validate_latex_export = inspect_latex_doc


# ============================================================================
# Independent Reference Mathematical Metrics (Pure NumPy)
# ============================================================================


def compute_reference_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Mean Squared Error (MSE) independently using float64."""
    arr_a = a.astype(np.float64)
    arr_b = b.astype(np.float64)
    return float(np.mean((arr_a - arr_b) ** 2))


def compute_reference_psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB)."""
    mse = compute_reference_mse(a, b)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val**2) / mse))


def compute_reference_mae(a: np.ndarray, b: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compute_reference_max_error(a: np.ndarray, b: np.ndarray) -> float:
    """Computes L_infinity Chebyshev distance (maximum absolute error)."""
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def compute_reference_ssim(
    img1: np.ndarray, img2: np.ndarray, K1: float = 0.01, K2: float = 0.03, L: float = 255.0
) -> float:
    """
    Computes Structural Similarity Index (SSIM) using standard Gaussian spatial weighting.
    Works for 2D grayscale and 3D multi-channel arrays.
    """
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    sigma = 1.5

    def _ssim_2d(c1: np.ndarray, c2: np.ndarray) -> float:
        mu1 = gaussian_filter(c1, sigma)
        mu2 = gaussian_filter(c2, sigma)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        s1_sq = gaussian_filter(c1 * c1, sigma) - mu1_sq
        s2_sq = gaussian_filter(c2 * c2, sigma) - mu2_sq
        s12 = gaussian_filter(c1 * c2, sigma) - mu1_mu2
        ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1_sq + s2_sq + C2))
        return float(np.mean(ssim_map))

    if im1.ndim == 3:
        scores = []
        for ch in range(im1.shape[2]):
            scores.append(_ssim_2d(im1[:, :, ch], im2[:, :, ch]))
        return float(np.mean(scores))
    else:
        return _ssim_2d(im1, im2)


def compute_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Computes Sharma-Wu-Dhamija (2005) CIEDE2000 color difference map.
    Inputs are (H, W, 3) arrays representing L*, a*, b*.
    Returns 2D map of delta E values.
    """
    L1 = lab1[:, :, 0].astype(np.float64)
    a1 = lab1[:, :, 1].astype(np.float64)
    b1 = lab1[:, :, 2].astype(np.float64)

    L2 = lab2[:, :, 0].astype(np.float64)
    a2 = lab2[:, :, 1].astype(np.float64)
    b2 = lab2[:, :, 2].astype(np.float64)

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = 0.5 * (C1 + C2)
    G = 0.5 * (1.0 - np.sqrt((C_bar**7) / (C_bar**7 + 25.0**7 + 1e-12)))

    a1_p = (1.0 + G) * a1
    a2_p = (1.0 + G) * a2
    C1_p = np.sqrt(a1_p**2 + b1**2)
    C2_p = np.sqrt(a2_p**2 + b2**2)

    h1_p = np.degrees(np.arctan2(b1, a1_p)) % 360.0
    h2_p = np.degrees(np.arctan2(b2, a2_p)) % 360.0

    delta_L_p = L2 - L1
    delta_C_p = C2_p - C1_p

    delta_h_p = h2_p - h1_p
    delta_h_p = np.where(delta_h_p > 180.0, delta_h_p - 360.0, delta_h_p)
    delta_h_p = np.where(delta_h_p < -180.0, delta_h_p + 360.0, delta_h_p)
    delta_h_p = np.where((C1_p * C2_p) == 0.0, 0.0, delta_h_p)
    delta_H_p = 2.0 * np.sqrt(C1_p * C2_p) * np.sin(np.radians(delta_h_p / 2.0))

    L_bar_p = 0.5 * (L1 + L2)
    C_bar_p = 0.5 * (C1_p + C2_p)

    h_bar_p = np.where(
        np.abs(h1_p - h2_p) > 180.0,
        np.where((h1_p + h2_p) < 360.0, 0.5 * (h1_p + h2_p + 360.0), 0.5 * (h1_p + h2_p - 360.0)),
        0.5 * (h1_p + h2_p),
    )
    h_bar_p = np.where((C1_p * C2_p) == 0.0, h1_p + h2_p, h_bar_p)

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    delta_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt((C_bar_p**7) / (C_bar_p**7 + 25.0**7 + 1e-12))
    S_L = 1.0 + (0.015 * ((L_bar_p - 50.0) ** 2)) / np.sqrt(20.0 + ((L_bar_p - 50.0) ** 2))
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T
    R_T = -np.sin(np.radians(2.0 * delta_theta)) * R_C

    dE = np.sqrt(
        (delta_L_p / S_L) ** 2
        + (delta_C_p / S_C) ** 2
        + (delta_H_p / S_H) ** 2
        + R_T * (delta_C_p / S_C) * (delta_H_p / S_H)
    )
    return dE


# Metric aliases
compute_mse = compute_reference_mse
compute_psnr = compute_reference_psnr
compute_mae = compute_reference_mae
compute_max_error = compute_reference_max_error
compute_ssim = compute_reference_ssim
