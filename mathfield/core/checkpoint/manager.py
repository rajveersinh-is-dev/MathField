"""
mathfield/core/checkpoint/manager.py

Robust Atomic Checkpoint Manager for MathField.
Manages disk persistence of .mfchk checkpoint archives via Win32 atomic
two-phase commit (tempfile -> fsync -> os.replace), schema validation,
checkpoint rollback, history rotation, and pipeline context resumption.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mathfield.core.checkpoint.serializer import (
    SCHEMA_VERSION,
    CheckpointSerializer,
)


class CheckpointCorruptError(ValueError):
    """Raised when a checkpoint file is missing, truncated, unparseable, or schema-incompatible."""


class CheckpointManager:
    """
    High-level manager for saving, loading, rolling back, and verifying .mfchk checkpoints.
    Supports atomic two-phase commit, history retention, and cooperative execution controls.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        max_history: int = 5,
        default_extension: str = ".mfchk",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path.cwd()
        self.max_history = int(max_history)
        self.default_extension = default_extension if default_extension.startswith(".") else f".{default_extension}"
        self._paused: bool = False
        self._cancelled: bool = False

    def _resolve_path(self, path_or_name: Union[str, Path]) -> Path:
        p = Path(path_or_name)
        if not p.is_absolute():
            p = self.checkpoint_dir / p
        if p.suffix == "":
            p = p.with_suffix(self.default_extension)
        return p

    def pause(self) -> None:
        """Sets cooperative execution pause flag."""
        self._paused = True

    def resume(self) -> None:
        """Clears cooperative execution pause flag."""
        self._paused = False

    def cancel(self) -> None:
        """Sets cooperative execution cancel flag."""
        self._cancelled = True

    def is_paused(self) -> bool:
        """Returns True if paused."""
        return self._paused

    def is_cancelled(self) -> bool:
        """Returns True if cancelled."""
        return self._cancelled

    def save(
        self,
        arg1: Union[str, Path, Dict[str, Any]],
        arg2: Union[str, Path, Dict[str, Any]],
        create_backup: bool = True,
    ) -> str:
        """
        Save checkpoint payload using atomic two-phase commit:
        1. Write to temporary file (.tmp) in the exact same directory.
        2. Flush and fsync to guarantee physical disk synchronization.
        3. If target file already exists and create_backup is True, create .bak copy.
        4. Atomically rename/replace .tmp to target path using os.replace (MoveFileExW on Windows).

        Supports both save(path, payload) and save(payload, path) argument orders.
        """
        if isinstance(arg1, dict) and not isinstance(arg2, dict):
            payload = arg1
            path_or_name = arg2
        else:
            path_or_name = arg1
            payload = arg2

        target = self._resolve_path(path_or_name)
        target.parent.mkdir(parents=True, exist_ok=True)

        full_payload = dict(payload)
        if "schema_version" not in full_payload and "version" not in full_payload:
            full_payload["schema_version"] = SCHEMA_VERSION
        if "timestamp" not in full_payload:
            full_payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if "hardware_telemetry" not in full_payload:
            full_payload["hardware_telemetry"] = CheckpointSerializer.collect_hardware_telemetry()

        tmp_filename = f"{target.stem}_{uuid.uuid4().hex[:8]}.tmp"
        tmp_path = target.parent / tmp_filename

        try:
            json_text = json.dumps(full_payload, indent=2, allow_nan=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json_text)
                f.flush()
                os.fsync(f.fileno())

            # Create backup if destination already exists
            if create_backup and target.exists():
                bak_path = target.with_suffix(f"{target.suffix}.bak")
                try:
                    shutil.copy2(target, bak_path)
                except Exception:
                    pass

            # Atomic replace with transient lock retry loop for Windows
            last_err: Optional[Exception] = None
            for attempt in range(5):
                try:
                    os.replace(tmp_path, target)
                    last_err = None
                    break
                except PermissionError as pe:
                    last_err = pe
                    time.sleep(0.025 * (2**attempt))

            if last_err is not None:
                raise last_err

        except Exception as e:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            raise e

        return str(target)

    def load(self, path_or_name: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and validate .mfchk checkpoint file.
        Raises CheckpointCorruptError on missing file, corrupted JSON, or incompatible schema version.
        """
        path = self._resolve_path(path_or_name)
        if not path.is_file():
            raise CheckpointCorruptError(f"Checkpoint file does not exist: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise CheckpointCorruptError(f"Checkpoint JSON corrupted: {err}") from err

        if not isinstance(data, dict):
            raise CheckpointCorruptError("Checkpoint contents must be a JSON dictionary.")

        # Schema version check
        ver_str = data.get("schema_version") or data.get("version")
        if ver_str is not None:
            major_ver = str(ver_str).split(".")[0]
            curr_major = SCHEMA_VERSION.split(".")[0]
            if major_ver != curr_major:
                raise CheckpointCorruptError(
                    f"Incompatible checkpoint schema version: {ver_str} (expected major version {curr_major})"
                )

        return data

    def rollback(self, path_or_name: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Restore prior valid checkpoint from its .bak backup copy and return restored payload.
        """
        if path_or_name is not None:
            target = self._resolve_path(path_or_name)
        else:
            # Find latest checkpoint with a .bak file in checkpoint_dir
            candidates = sorted(
                self.checkpoint_dir.glob(f"*{self.default_extension}.bak"), key=lambda p: p.stat().st_mtime
            )
            if not candidates:
                raise FileNotFoundError(f"No backup checkpoints found in {self.checkpoint_dir}")
            bak_path = candidates[-1]
            target = bak_path.with_suffix("")
            os.replace(bak_path, target)
            return self.load(target)

        bak_path = target.with_suffix(f"{target.suffix}.bak")
        if not bak_path.is_file():
            raise FileNotFoundError(f"No backup checkpoint found for rollback at {bak_path}")

        os.replace(bak_path, target)
        return self.load(target)

    def delete(self, path_or_name: Union[str, Path]) -> bool:
        """Deletes checkpoint and associated .bak backup file."""
        target = self._resolve_path(path_or_name)
        deleted = False
        if target.exists():
            target.unlink()
            deleted = True
        bak = target.with_suffix(f"{target.suffix}.bak")
        if bak.exists():
            bak.unlink()
            deleted = True
        return deleted

    def list_checkpoints(self, prefix: Optional[str] = None) -> List[str]:
        """
        Enumerate valid checkpoints in checkpoint_dir, sorted chronologically.
        Returns list of file path strings.
        """
        if not self.checkpoint_dir.exists():
            return []

        pattern = f"{prefix}*{self.default_extension}" if prefix else f"*{self.default_extension}"
        files = sorted(self.checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        return [str(p) for p in files]

    def rotate_checkpoints(self, prefix: Optional[str] = None, max_keep: Optional[int] = None) -> List[str]:
        """
        Prune oldest checkpoints exceeding max_keep retention quota.
        """
        keep_count = max_keep if max_keep is not None else self.max_history
        all_chk = self.list_checkpoints(prefix=prefix)
        if len(all_chk) <= keep_count:
            return []

        to_remove = list(all_chk) if keep_count <= 0 else all_chk[:-keep_count]
        removed_paths = []
        for path_str in to_remove:
            p = Path(path_str)
            try:
                self.delete(p)
                removed_paths.append(str(p))
            except Exception:
                pass
        return removed_paths

    def save_from_context(
        self,
        context: Any,
        path_or_name: Union[str, Path],
        stage: str = "STAGE_4_OPTIMIZATION",
        iteration: int = 0,
        max_iter: int = 100,
        loss: Optional[float] = None,
        mode: str = "discovery",
        rng: Optional[Any] = None,
    ) -> str:
        """
        Extract full state from PipelineContext and serialize to disk.
        """
        payload = CheckpointSerializer.serialize_context(
            context=context,
            stage=stage,
            iteration=iteration,
            max_iter=max_iter,
            loss=loss,
            mode=mode,
            rng=rng,
        )
        return self.save(path_or_name, payload)

    def load_into_context(self, path_or_name: Union[str, Path], context: Any) -> Dict[str, Any]:
        """
        Load checkpoint data and inject restored state into an existing PipelineContext.
        """
        data = self.load(path_or_name)
        CheckpointSerializer.deserialize_context(data, context)
        return data
