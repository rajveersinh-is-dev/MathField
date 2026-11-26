"""
mathfield/core/checkpoint/serializer.py

Bit-Exact Checkpoint Serialization Engine for MathField.
Provides IEEE-754 hex float precision preservation, modern NumPy BitGenerator
state serialization, AST JSON serialization, hardware telemetry capture,
and pipeline context checkpointing.
"""

from __future__ import annotations

import platform
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from mathfield.core.ast.nodes import ASTNode
from mathfield.core.model_codec import (
    deserialize_mode_b_model,
    is_mode_b_global_model,
    serialize_mode_b_model,
)

SCHEMA_VERSION: str = "1.0.0"
SUPPORTED_BIT_GENERATORS = {"PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937"}


class CheckpointSerializer:
    """
    Serializes and deserializes reconstruction checkpoints with bit-level fidelity.
    """

    @staticmethod
    def serialize_float(val: float) -> str:
        """
        Serialize a float to exact IEEE-754 hexadecimal representation.
        Handles standard numbers, -0.0, subnormals, infinities, and NaN.
        """
        f_val = float(val)
        return f_val.hex()

    @staticmethod
    def deserialize_float(hex_str: str) -> float:
        """
        Deserialize an IEEE-754 hexadecimal string back to float with zero precision loss.
        """
        s = str(hex_str).strip()
        return float.fromhex(s)

    @classmethod
    def serialize_parameter(cls, val: float, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Format a single parameter value with exact hex, high-precision decimal, and raw float.
        """
        f_val = float(val)
        return {
            "float": f_val,
            "hex": cls.serialize_float(f_val),
            "decimal": f"{f_val:.17g}",
        }

    @classmethod
    def serialize_parameters(cls, params: Union[Dict[str, Any], np.ndarray, Sequence[float]]) -> Dict[str, Any]:
        """
        Serialize parameter collection (dict, array, or sequence) to named hex parameter entries.
        Guarantees shape and dtype preservation for ndarrays.
        """
        if isinstance(params, np.ndarray):
            arr = params
            out: Dict[str, Any] = {
                "__type__": "ndarray",
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
                "hex": [float(x).hex() for x in arr.flat],
                "decimal": [f"{float(x):.17g}" for x in arr.flat],
            }
            # Populate indexed parameter entries c0, c1...
            for i, v in enumerate(arr.flat):
                out[f"c{i}"] = {
                    "float": float(v),
                    "hex": float(v).hex(),
                    "decimal": f"{float(v):.17g}",
                }
            return out

        elif isinstance(params, dict):
            # Mark named dictionaries explicitly so deserialization can
            # preserve names instead of silently collapsing them to an array.
            out: Dict[str, Any] = {"__type__": "named_dict"}
            for k, v in params.items():
                if isinstance(v, dict) and "hex" in v:
                    out[str(k)] = v
                else:
                    f_val = float(v)
                    out[str(k)] = {
                        "float": f_val,
                        "hex": cls.serialize_float(f_val),
                        "decimal": f"{f_val:.17g}",
                    }
            return out

        elif isinstance(params, (list, tuple)):
            arr = np.asarray(params, dtype=np.float64)
            return cls.serialize_parameters(arr)

        else:
            raise TypeError(f"Unsupported parameter container type: {type(params)}")

    @classmethod
    def deserialize_parameters(cls, data: Dict[str, Any]) -> Union[np.ndarray, Dict[str, float]]:
        """
        Deserialize parameters back to an exact numpy ndarray, or to an
        ordered dict of floats for named parameter collections.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data)}")

        if data.get("__type__") == "named_dict":
            named: Dict[str, float] = {}
            for key, value in data.items():
                if key.startswith("__"):
                    continue
                if isinstance(value, dict) and "hex" in value:
                    named[key] = cls.deserialize_float(value["hex"])
                elif isinstance(value, (int, float)):
                    named[key] = float(value)
                elif isinstance(value, str):
                    named[key] = cls.deserialize_float(value)
            return named

        if "shape" in data and "hex" in data:
            hex_list = data["hex"]
            shape = tuple(data["shape"])
            dtype = np.dtype(data.get("dtype", np.float64))
            arr = np.array([cls.deserialize_float(h) for h in hex_list], dtype=dtype)
            return arr.reshape(shape)

        # Fallback for dictionaries with parameter keys (e.g. c0, c1...)
        # Sort keys by integer index if format is c<int>, else natural sort
        def key_sort(k: str) -> Tuple[int, Union[int, str]]:
            if k.startswith("c") and k[1:].isdigit():
                return (0, int(k[1:]))
            return (1, k)

        param_keys = [k for k in data if not k.startswith("__") and k not in ("shape", "dtype", "decimal", "hex")]
        param_keys.sort(key=key_sort)

        values = []
        for k in param_keys:
            v = data[k]
            if isinstance(v, dict) and "hex" in v:
                values.append(cls.deserialize_float(v["hex"]))
            elif isinstance(v, (int, float)):
                values.append(float(v))
            elif isinstance(v, str):
                values.append(cls.deserialize_float(v))
            else:
                raise ValueError(f"Unparseable parameter entry for {k}: {v}")

        return np.array(values, dtype=np.float64)

    @classmethod
    def serialize_rng_state(cls, rng_or_state: Union[np.random.Generator, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recursively serialize a NumPy BitGenerator state dictionary into JSON-serializable types.
        Converts all internal NumPy ndarrays (counters, keys, buffers) into typed dict structures.
        """
        if isinstance(rng_or_state, np.random.Generator) or hasattr(rng_or_state, "bit_generator"):
            raw_state = rng_or_state.bit_generator.state
        elif isinstance(rng_or_state, dict):
            raw_state = rng_or_state
        else:
            raise TypeError(f"Expected np.random.Generator or dict, got {type(rng_or_state)}")

        def _encode_obj(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _encode_obj(v) for k, v in obj.items()}
            elif isinstance(obj, np.ndarray):
                return {
                    "__ndarray__": True,
                    "dtype": str(obj.dtype),
                    "shape": list(obj.shape),
                    "data": obj.tolist(),
                }
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            return obj

        return _encode_obj(raw_state)

    @classmethod
    def deserialize_rng_state(cls, state_dict: Dict[str, Any]) -> np.random.Generator:
        """
        Deserialize state dictionary and reconstruct an active np.random.Generator
        with identical internal bit generator sequence state across PCG64, Philox, SFC64, MT19937.
        """

        def _decode_obj(obj: Any) -> Any:
            if isinstance(obj, dict):
                if obj.get("__ndarray__") is True:
                    arr = np.array(obj["data"], dtype=np.dtype(obj["dtype"]))
                    if "shape" in obj:
                        arr = arr.reshape(obj["shape"])
                    return arr
                return {k: _decode_obj(v) for k, v in obj.items()}
            return obj

        clean_state = _decode_obj(state_dict)
        bg_name = clean_state.get("bit_generator")
        if not bg_name:
            raise ValueError("State dict missing 'bit_generator' descriptor.")

        if bg_name not in SUPPORTED_BIT_GENERATORS:
            raise ValueError(f"Unsupported BitGenerator: {bg_name}. Must be in {SUPPORTED_BIT_GENERATORS}")

        bg_class = getattr(np.random, bg_name, None)
        if bg_class is None:
            raise ValueError(f"NumPy does not expose bit generator: {bg_name}")

        bg = bg_class()
        bg.state = clean_state
        return np.random.Generator(bg)

    @staticmethod
    def serialize_ast(node: ASTNode) -> Dict[str, Any]:
        """Serialize an AST node expression tree to dictionary via to_dict()."""
        return node.to_dict()

    @staticmethod
    def deserialize_ast(ast_dict: Dict[str, Any]) -> ASTNode:
        """Reconstruct an ASTNode expression tree from dictionary via ASTNode.from_dict()."""
        return ASTNode.from_dict(ast_dict)

    @classmethod
    def collect_hardware_telemetry(cls) -> Dict[str, Any]:
        """
        Gather non-invasive system telemetry at checkpoint creation time.
        """
        telemetry: Dict[str, Any] = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            import psutil

            process = psutil.Process()
            telemetry["cpu_percent"] = psutil.cpu_percent(interval=None)
            telemetry["process_ram_rss_mb"] = round(process.memory_info().rss / (1024.0 * 1024.0), 2)
            telemetry["system_ram_available_mb"] = round(psutil.virtual_memory().available / (1024.0 * 1024.0), 2)
        except Exception:
            pass

        try:
            import cupy

            if cupy.cuda.is_available():
                props = cupy.cuda.runtime.getDeviceProperties(0)
                mem_info = cupy.cuda.runtime.memGetInfo()
                telemetry["gpu_name"] = props["name"].decode("utf-8")
                telemetry["gpu_vram_free_mb"] = round(mem_info[0] / (1024.0 * 1024.0), 2)
                telemetry["gpu_vram_total_mb"] = round(mem_info[1] / (1024.0 * 1024.0), 2)
        except Exception:
            pass

        return telemetry

    @classmethod
    def serialize_context(
        cls,
        context: Any,
        stage: str = "STAGE_4_OPTIMIZATION",
        iteration: int = 0,
        max_iter: int = 100,
        loss: Optional[float] = None,
        mode: str = "discovery",
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """
        Helper extracting full optimization state from a PipelineContext object.
        """
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "iteration": iteration,
            "max_iter": max_iter,
            "loss": loss if loss is not None else context.metrics.get("loss", 0.0),
            "mode": mode,
            "coordinate_domain": context.domain.to_dict() if hasattr(context.domain, "to_dict") else {},
            "metrics": dict(context.metrics) if context.metrics else {},
            "expressions": {},
            "parameters": {},
            "ast_trees": {},
            "hardware_telemetry": cls.collect_hardware_telemetry(),
        }

        if context.global_model and isinstance(context.global_model, dict):
            if is_mode_b_global_model(context.global_model):
                image = np.asarray(context.image)
                channel_count = image.shape[2] if image.ndim == 3 else 1
                default_labels = {1: ["Y"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}
                channel_labels = default_labels.get(channel_count)
                config = context.config if isinstance(context.config, dict) else {}
                local_decomposition = context.local_decomposition
                has_partition = isinstance(local_decomposition, dict) and any(
                    key in local_decomposition for key in ("partition_of_unity", "partition", "pou")
                )
                partition = None
                if not has_partition:
                    partition = {
                        "overlap_ratio": float(config.get("overlap_ratio", 0.20)),
                        "separable": bool(config.get("separable", True)),
                    }
                payload["mode_b_model"] = serialize_mode_b_model(
                    context.global_model,
                    local_decomposition,
                    channel_labels=channel_labels,
                    partition=partition,
                )
            else:
                if "symbolic_ast" in context.global_model and hasattr(context.global_model["symbolic_ast"], "to_dict"):
                    ast_node = context.global_model["symbolic_ast"]
                    payload["ast_trees"]["symbolic_ast"] = cls.serialize_ast(ast_node)
                    payload["expressions"]["symbolic"] = str(ast_node)
                if "channel_asts" in context.global_model:
                    for idx, ast_node in enumerate(context.global_model["channel_asts"]):
                        payload["ast_trees"][f"channel_{idx}"] = cls.serialize_ast(ast_node)
                        payload["expressions"][f"channel_{idx}"] = str(ast_node)

        if rng is not None:
            payload["rng_state"] = cls.serialize_rng_state(rng)

        return payload

    @classmethod
    def deserialize_context(cls, data: Dict[str, Any], context: Any) -> Any:
        """
        Helper injecting deserialized checkpoint data into a PipelineContext object.
        """
        if "metrics" in data and isinstance(data["metrics"], dict):
            context.metrics.update(data["metrics"])

        if data.get("mode_b_model") is not None:
            global_model, local_decomposition = deserialize_mode_b_model(data["mode_b_model"])
            context.global_model = global_model
            context.local_decomposition = local_decomposition

        if "ast_trees" in data and isinstance(data["ast_trees"], dict):
            if context.global_model is None:
                context.global_model = {}
            for k, tree_dict in data["ast_trees"].items():
                context.global_model[k] = cls.deserialize_ast(tree_dict)

        return context
