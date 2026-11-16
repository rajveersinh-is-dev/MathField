"""
tests/test_checkpoint.py

Comprehensive Unit & Integration Test Suite for Milestone 3 Checkpointing:
- Bit-exact IEEE-754 hex float serialization roundtrip (subnormals, infinities, NaNs, -0.0).
- AST serialization roundtrip (complex trees, structural equality, hash equality, L_inf = 0.0).
- BitGenerator state serialization for modern PRNGs (PCG64, Philox, SFC64, MT19937).
- Atomic two-phase write safety: crash/error simulation and destination file integrity.
- Checkpoint corruption detection, version mismatch rejection, and history rollback.
- Pipeline resume parity: bit-exact optimization trajectory continuation.
"""

import json
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)
from mathfield.core.ast.parser import parse
from mathfield.core.checkpoint.manager import (
    CheckpointCorruptError,
    CheckpointManager,
)
from mathfield.core.checkpoint.serializer import CheckpointSerializer


class TestHexFloatExactness:
    """Tests for IEEE-754 hex float bit-exact preservation."""

    @pytest.mark.parametrize(
        "val",
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.125,
            1.0 / 3.0,
            math.pi,
            math.e,
            math.sqrt(2.0),
            0.1,
            0.7,
            1e-300,
            1e300,
            float.fromhex("0x1.0000000000001p-1022"),  # subnormal
            float.fromhex("0x0.0000000000001p-1022"),  # smallest positive subnormal
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
    )
    def test_hex_float_roundtrip_individual(self, val: float) -> None:
        """TC-CHK-01: Verify IEEE-754 hex float serialization round-trips with zero precision loss."""
        hex_str = CheckpointSerializer.serialize_float(val)
        recovered = CheckpointSerializer.deserialize_float(hex_str)

        if math.isnan(val):
            assert math.isnan(recovered)
        else:
            assert val == recovered
            assert float(val).hex() == float(recovered).hex()
            assert math.copysign(1.0, val) == math.copysign(1.0, recovered)

    def test_parameter_array_hex_serialization_roundtrip(self) -> None:
        """TC-CHK-02: Verify 1D and 2D parameter ndarrays maintain bitwise equality via hex serialization."""
        params = np.array([
            0.0,
            -0.0,
            math.pi,
            1.0 / 3.0,
            float.fromhex("0x1.0000000000001p-1022"),
            1e300,
        ], dtype=np.float64)

        serialized = CheckpointSerializer.serialize_parameters(params)
        json_str = json.dumps(serialized)
        restored = CheckpointSerializer.deserialize_parameters(json.loads(json_str))

        assert restored.shape == params.shape
        assert restored.dtype == params.dtype
        np.testing.assert_array_equal(restored, params)
        for orig, rest in zip(params, restored):
            assert float(orig).hex() == float(rest).hex()


class TestASTSerialization:
    """Tests for AST serialization, hash preservation, and evaluation parity."""

    @pytest.mark.parametrize(
        "expr_str",
        [
            "x + y",
            "sin(2.5 * x) + cos(1.8 * y)",
            "c0 * exp(-c1 * (x^2 + y^2)) + c2 * tanh(c3 * x)",
            "(x^2 - y^2) / (x^2 + y^2 + 0.05)",
            "clamp(x * 2.0, -0.5, 0.5) + abs(y) + sqrt(abs(x) + 0.01)",
            "rbf_gaussian(x, 1.5) + rbf_wendland(y, 2.0)",
        ],
    )
    def test_ast_serialization_roundtrip_and_parity(self, expr_str: str) -> None:
        """TC-CHK-03: Verify AST JSON serialization maintains tree equality, hash equality, and L_inf = 0.0."""
        tree = parse(expr_str)
        ast_dict = CheckpointSerializer.serialize_ast(tree)
        json_str = json.dumps(ast_dict)
        restored = CheckpointSerializer.deserialize_ast(json.loads(json_str))

        assert restored == tree
        assert hash(restored) == hash(tree)

        # Numerical evaluation parity on a 50x50 mesh
        x = np.linspace(-1.0, 1.0, 50)
        X, Y = np.meshgrid(x, x)
        params = np.array([1.2, 0.5, -3.1, 0.8, 2.0])

        eval_orig = tree.compile_numpy()(X, Y, params)
        eval_rest = restored.compile_numpy()(X, Y, params)
        linf = float(np.max(np.abs(eval_orig - eval_rest)))
        assert linf == 0.0

    def test_deep_ast_tree_serialization(self) -> None:
        """TC-CHK-04: Verify deeply nested AST (depth 12) serializes and reconstructs without stack overflow."""
        node: ASTNode = VariableNode("x")
        for _ in range(12):
            node = BinaryOpNode("+", node, FunctionCallNode("sin", (VariableNode("y"),)))

        ast_dict = CheckpointSerializer.serialize_ast(node)
        restored = CheckpointSerializer.deserialize_ast(json.loads(json.dumps(ast_dict)))
        assert restored == node
        assert restored.complexity() == node.complexity()


class TestBitGeneratorStateSerialization:
    """Tests for PRNG BitGenerator state persistence across modern engines."""

    @pytest.mark.parametrize(
        "bitgen_cls",
        [
            np.random.PCG64,
            np.random.Philox,
            np.random.SFC64,
            np.random.MT19937,
        ],
    )
    def test_bitgenerator_sequence_continuation(self, bitgen_cls) -> None:
        """TC-CHK-05: Verify exact RNG continuation across PCG64, Philox, SFC64, and MT19937."""
        rng = np.random.Generator(bitgen_cls(123456))
        # Warmup / advance state
        _ = rng.standard_normal(100)

        # Serialize state and guarantee JSON serializability
        state_dict = CheckpointSerializer.serialize_rng_state(rng)
        json_text = json.dumps(state_dict)
        unpacked_dict = json.loads(json_text)

        # Ground truth continuation on original generator
        expected_draws = rng.standard_normal(500)

        # Restore state into new generator
        resumed_rng = CheckpointSerializer.deserialize_rng_state(unpacked_dict)
        actual_draws = resumed_rng.standard_normal(500)

        # Assert bit-exact array equality
        np.testing.assert_array_equal(actual_draws, expected_draws)


class TestAtomicWriteSafety:
    """Tests for two-phase atomic write, crash simulation, and prior file integrity."""

    def test_atomic_write_success(self, tmp_path: Path) -> None:
        """TC-CHK-06: Verify atomic two-phase write produces valid file and cleans up .tmp files."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        target = tmp_path / "model.mfchk"
        payload = {
            "schema_version": "1.0.0",
            "stage": "STAGE_4_OPTIMIZATION",
            "iteration": 25,
            "parameters": {"c0": {"float": 1.5, "hex": float(1.5).hex()}},
        }
        mgr.save(target, payload)

        assert target.exists()
        # Verify no dangling temporary files
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

        # Load and verify
        loaded = mgr.load(target)
        assert loaded["iteration"] == 25

    def test_atomic_write_crash_simulation_preserves_original(self, tmp_path: Path) -> None:
        """TC-CHK-07: Verify crash during write leaves prior checkpoint completely intact."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        target = tmp_path / "safe.mfchk"

        # Version 1 (valid)
        v1_payload = {"schema_version": "1.0.0", "stage": "STAGE_1", "iteration": 5}
        mgr.save(target, v1_payload)
        original_content = target.read_text(encoding="utf-8")

        # Simulate crash during write of Version 2
        with mock.patch("os.fsync", side_effect=IOError("Simulated disk write failure")):
            with pytest.raises(IOError):
                mgr.save(target, {"schema_version": "1.0.0", "stage": "STAGE_2", "iteration": 10})

        # Destination file must be completely unchanged
        assert target.read_text(encoding="utf-8") == original_content

        # No temporary files left behind
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestCorruptionAndVersionHandling:
    """Tests for truncated JSON, garbage files, and future version rejection."""

    def test_corrupted_json_rejection(self, tmp_path: Path) -> None:
        """TC-CHK-08: Verify truncated or malformed JSON raises CheckpointCorruptError."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        bad_file = tmp_path / "corrupt.mfchk"
        bad_file.write_text('{"schema_version": "1.0.0", "stage": "STAGE_4", "params": {', encoding="utf-8")

        with pytest.raises(CheckpointCorruptError):
            mgr.load(bad_file)

    def test_binary_garbage_rejection(self, tmp_path: Path) -> None:
        """TC-CHK-09: Verify binary garbage raises CheckpointCorruptError."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        bad_file = tmp_path / "garbage.mfchk"
        bad_file.write_bytes(b"\x00\xff\xfe\x01\x88\x99\xaa\xbb")

        with pytest.raises(CheckpointCorruptError):
            mgr.load(bad_file)

    def test_incompatible_version_rejection(self, tmp_path: Path) -> None:
        """TC-CHK-10: Verify incompatible schema version (e.g. 99.0.0) raises CheckpointCorruptError."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        future_file = tmp_path / "future.mfchk"
        future_file.write_text(json.dumps({"schema_version": "99.0.0", "stage": "STAGE_1", "iteration": 0}), encoding="utf-8")

        with pytest.raises(CheckpointCorruptError, match="version"):
            mgr.load(future_file)


class TestRollbackAndRotation:
    """Tests for checkpoint rollback and history retention."""

    def test_checkpoint_rollback(self, tmp_path: Path) -> None:
        """TC-CHK-11: Verify rollback restores prior checkpoint state from .bak."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        target = tmp_path / "rollback_test.mfchk"

        # Save initial state
        mgr.save(target, {"schema_version": "1.0.0", "iteration": 10, "loss": 0.5})
        assert target.exists()

        # Save updated state (creates .bak of iteration 10)
        mgr.save(target, {"schema_version": "1.0.0", "iteration": 20, "loss": 0.2})
        loaded_v2 = mgr.load(target)
        assert loaded_v2["iteration"] == 20

        # Rollback
        restored = mgr.rollback(target)
        assert restored["iteration"] == 10
        assert restored["loss"] == 0.5

    def test_checkpoint_rotation(self, tmp_path: Path) -> None:
        """TC-CHK-12: Verify rotate_checkpoints prunes older checkpoints beyond retention limit."""
        mgr = CheckpointManager(checkpoint_dir=tmp_path, max_history=3)
        for i in range(5):
            mgr.save(tmp_path / f"ckpt_{i:02d}.mfchk", {"iteration": i})

        checkpoints = mgr.list_checkpoints()
        assert len(checkpoints) == 5

        pruned = mgr.rotate_checkpoints(max_keep=3)
        assert len(pruned) == 2

        remaining = mgr.list_checkpoints()
        assert len(remaining) == 3


class TestPipelineResumeParity:
    """End-to-end tests for exact optimization resumption from checkpoint."""

    def test_pipeline_resume_bit_exact_parity(self, tmp_path: Path) -> None:
        """TC-CHK-13: Verify resuming optimization from checkpoint reproduces identical bit-exact trajectory."""
        chk_file = tmp_path / "resume_test.mfchk"

        # Mock optimization task: gradient descent on quadratic loss
        # Target: minimize f(w) = 0.5 * w^T A w - b^T w
        A = np.array([[3.0, 1.0], [1.0, 2.0]], dtype=np.float64)
        b = np.array([1.5, -0.5], dtype=np.float64)
        lr = 0.1

        # 1. Uninterrupted run for 20 iterations
        w_uninterrupted = np.array([5.0, -3.0], dtype=np.float64)
        trajectory_uninterrupted = [w_uninterrupted.copy()]
        for _ in range(20):
            grad = A @ w_uninterrupted - b
            w_uninterrupted -= lr * grad
            trajectory_uninterrupted.append(w_uninterrupted.copy())

        # 2. Interrupted run: stop at iteration 10, save checkpoint
        w_interrupted = np.array([5.0, -3.0], dtype=np.float64)
        for _ in range(10):
            grad = A @ w_interrupted - b
            w_interrupted -= lr * grad

        mgr = CheckpointManager(checkpoint_dir=tmp_path)
        mgr.save(chk_file, {
            "schema_version": "1.0.0",
            "stage": "STAGE_4_OPTIMIZATION",
            "iteration": 10,
            "parameters": CheckpointSerializer.serialize_parameters(w_interrupted),
        })

        # 3. Resume from checkpoint at iteration 10
        loaded_data = mgr.load(chk_file)
        w_resumed = CheckpointSerializer.deserialize_parameters(loaded_data["parameters"])
        assert np.array_equal(w_resumed, w_interrupted)

        # Continue from iter 10 to 20
        trajectory_resumed = [w_resumed.copy()]
        for _ in range(10):
            grad = A @ w_resumed - b
            w_resumed -= lr * grad
            trajectory_resumed.append(w_resumed.copy())

        # Assert final parameters are bit-identical (L_inf = 0.0)
        np.testing.assert_array_equal(w_resumed, w_uninterrupted)
        linf = float(np.max(np.abs(w_resumed - w_uninterrupted)))
        assert linf == 0.0
