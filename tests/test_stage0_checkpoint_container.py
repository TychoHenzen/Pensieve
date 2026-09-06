"""Contracts for the bounded Stage 0 checkpoint ZIP container."""

from __future__ import annotations

import importlib
import json
import pickle
import struct
import warnings
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from train import run_alternating
from train.alternating_checkpoint import CheckpointSchedule
from train.alternating_scheduler import EvaluationRecord
from train.training_results import EvaluationResult, ExperimentPosition, StepResult

METADATA_MEMBER = "metadata.json"
TENSORS_MEMBER = "tensors.safetensors"


def _checkpoint_module():
    """Import the public checkpoint-container API under test."""

    return importlib.import_module("train.stage0_checkpoint")


def _metadata_bytes(*, schema_version: int = 2, extra_json: str = "") -> bytes:
    suffix = f",{extra_json}" if extra_json else ""
    return f'{{"schema_version":{schema_version}{suffix}}}'.encode()


def _tiny_safetensors_bytes() -> bytes:
    """Return a valid safetensors payload containing one float32 scalar."""

    header = json.dumps(
        {
            "model.encoder.projection.bias": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, 4],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * (-len(header) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)


def _validated_tiny_metadata(*, shape: list[int] | None = None) -> SimpleNamespace:
    tensor = {
        "name": "model.encoder.projection.bias",
        "shape": shape or [1],
        "dtype": "float32",
        "role": "model_parameter",
    }
    return SimpleNamespace(
        tensor_manifest=(tensor,),
        to_dict=lambda: {"schema_version": 2},
    )


def _write_archive(
    path: Path,
    *,
    members: list[tuple[str, bytes, int]] | None = None,
) -> None:
    archive_members = members or [
        (METADATA_MEMBER, _metadata_bytes(), ZIP_STORED),
        (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
    ]
    with ZipFile(path, "w") as archive, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, payload, compression in archive_members:
            archive.writestr(name, payload, compress_type=compression)


def _read(path: Path, tensor_loader: Callable[[bytes], object] | None = None):
    module = _checkpoint_module()
    loader = tensor_loader or (lambda payload: {"payload_size": len(payload)})
    return module.read_checkpoint_container(path, tensor_loader=loader)


def _assert_rejected_before_tensor_load(
    path: Path,
    *,
    match: str,
) -> None:
    module = _checkpoint_module()
    calls: list[bytes] = []

    with pytest.raises(module.CheckpointContainerError, match=match):
        module.read_checkpoint_container(
            path,
            tensor_loader=calls.append,
        )

    assert calls == []


def test_writer_emits_v2_stored_archive_with_exact_bounded_members(tmp_path: Path) -> None:
    module = _checkpoint_module()
    path = tmp_path / "phase-7.ckpt"
    tensor_payload = _tiny_safetensors_bytes()

    with patch.object(
        module,
        "_canonical_metadata_bytes",
        return_value=(b'{"schema_version":2}', None),
    ):
        module.write_checkpoint_container(
            path,
            metadata={"schema_version": 2},
            tensor_payload=tensor_payload,
        )

    assert module.MAX_METADATA_BYTES == 1 * 1024 * 1024
    assert module.MAX_TENSOR_BYTES == 64 * 1024 * 1024
    with ZipFile(path, "r") as archive:
        members = archive.infolist()
        assert [member.filename for member in members] == [
            METADATA_MEMBER,
            TENSORS_MEMBER,
        ]
        assert all(member.compress_type == ZIP_STORED for member in members)
        assert members[0].file_size <= module.MAX_METADATA_BYTES
        assert members[1].file_size <= module.MAX_TENSOR_BYTES
        assert json.loads(archive.read(METADATA_MEMBER)) == {"schema_version": 2}
        assert archive.read(TENSORS_MEMBER) == tensor_payload


def test_writer_validation_failure_preserves_target_and_leaves_no_temp_file(
    tmp_path: Path,
) -> None:
    module = _checkpoint_module()
    path = tmp_path / "existing.ckpt"
    original = b"existing checkpoint bytes"
    path.write_bytes(original)

    with pytest.raises(module.CheckpointMetadataError):
        module.write_checkpoint_container(
            path,
            metadata={"schema_version": 3},
            tensor_payload=_tiny_safetensors_bytes(),
        )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_writer_replace_failure_preserves_target_and_removes_temp_file(
    tmp_path: Path,
) -> None:
    module = _checkpoint_module()
    path = tmp_path / "existing.ckpt"
    original = b"existing checkpoint bytes"
    path.write_bytes(original)

    with (
        patch.object(
            module,
            "_canonical_metadata_bytes",
            return_value=(b'{"schema_version":2}', None),
        ),
        patch.object(module.os, "replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        module.write_checkpoint_container(
            path,
            metadata={"schema_version": 2},
            tensor_payload=_tiny_safetensors_bytes(),
        )

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_reader_decodes_metadata_before_loading_safetensors(tmp_path: Path) -> None:
    module = _checkpoint_module()
    path = tmp_path / "epoch-3.ckpt"
    _write_archive(path)
    events: list[str] = []

    def validate(metadata: object):
        assert metadata == {"schema_version": 2}
        events.append("metadata")
        return _validated_tiny_metadata()

    with patch.object(module, "validate_checkpoint_metadata", side_effect=validate):
        loaded = module.read_checkpoint_container(
            path,
            tensor_loader=lambda payload: events.append("tensors") or {"weight": 1.0},
        )

    assert loaded.metadata == {"schema_version": 2}
    assert loaded.tensors == {"weight": 1.0}
    assert events == ["metadata", "tensors"]


def test_reader_uses_the_default_safetensors_cpu_loader(tmp_path: Path) -> None:
    module = _checkpoint_module()
    path = tmp_path / "cpu.ckpt"
    _write_archive(path)

    with patch.object(
        module,
        "validate_checkpoint_metadata",
        return_value=_validated_tiny_metadata(),
    ):
        loaded = module.read_checkpoint_container(path)

    tensor = loaded.tensors["model.encoder.projection.bias"]
    assert tensor.device.type == "cpu"
    assert tensor.shape == (1,)
    assert tensor.item() == 1.0


def test_reader_rejects_incomplete_metadata_before_reading_tensor_member(
    tmp_path: Path,
) -> None:
    module = _checkpoint_module()
    path = tmp_path / "incomplete-metadata.ckpt"
    _write_archive(path)
    member_reads: list[str] = []
    original_read = module._read_bounded_member

    def record_read(archive, info, maximum):
        member_reads.append(info.filename)
        return original_read(archive, info, maximum)

    with (
        patch.object(module, "_read_bounded_member", side_effect=record_read),
        pytest.raises(module.CheckpointMetadataError, match=r"\$\.identity"),
    ):
        module.read_checkpoint_container(path, tensor_loader=lambda _: None)

    assert member_reads == [METADATA_MEMBER]


def test_reader_rejects_tensor_shape_before_calling_loader(tmp_path: Path) -> None:
    module = _checkpoint_module()
    path = tmp_path / "shape-mismatch.ckpt"
    _write_archive(path)
    calls: list[bytes] = []

    with (
        patch.object(
            module,
            "validate_checkpoint_metadata",
            return_value=_validated_tiny_metadata(shape=[2]),
        ),
        pytest.raises(module.CheckpointContainerError, match=r"shape.*tensor_manifest"),
    ):
        module.read_checkpoint_container(
            path,
            tensor_loader=calls.append,
        )

    assert calls == []


def test_reader_rejects_unknown_schema_before_loading_tensors(tmp_path: Path) -> None:
    path = tmp_path / "unknown-version.ckpt"
    _write_archive(
        path,
        members=[
            (METADATA_MEMBER, _metadata_bytes(schema_version=3), ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)schema_version.*3")


# covers: train/stage0-training :: Stage 0 identity accompanies training artifacts :: Legacy artifact rejected
def test_reader_rejects_legacy_pickle_without_calling_torch_load(tmp_path: Path) -> None:
    path = tmp_path / "legacy.ckpt"
    path.write_bytes(pickle.dumps({"version": 1, "state": "legacy"}))
    module = _checkpoint_module()

    with (
        patch("torch.load", side_effect=AssertionError("torch.load is forbidden")) as torch_load,
        pytest.raises(module.CheckpointContainerError, match=r"(?i)zip|checkpoint"),
    ):
        module.read_checkpoint_container(path, tensor_loader=lambda _: None)

    torch_load.assert_not_called()


@pytest.mark.parametrize("duplicate_member", [METADATA_MEMBER, TENSORS_MEMBER])
def test_reader_rejects_duplicate_member_names_before_tensor_load(
    tmp_path: Path,
    duplicate_member: str,
) -> None:
    path = tmp_path / f"duplicate-{duplicate_member}.ckpt"
    duplicated_payload = _metadata_bytes() if duplicate_member == METADATA_MEMBER else _tiny_safetensors_bytes()
    _write_archive(
        path,
        members=[
            (METADATA_MEMBER, _metadata_bytes(), ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
            (duplicate_member, duplicated_payload, ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(
        path,
        match=rf"(?i)duplicate.*{duplicate_member}",
    )


@pytest.mark.parametrize("compressed_member", [METADATA_MEMBER, TENSORS_MEMBER])
def test_reader_rejects_compressed_members_before_tensor_load(
    tmp_path: Path,
    compressed_member: str,
) -> None:
    path = tmp_path / f"compressed-{compressed_member}.ckpt"
    _write_archive(
        path,
        members=[
            (
                METADATA_MEMBER,
                _metadata_bytes(),
                ZIP_DEFLATED if compressed_member == METADATA_MEMBER else ZIP_STORED,
            ),
            (
                TENSORS_MEMBER,
                _tiny_safetensors_bytes(),
                ZIP_DEFLATED if compressed_member == TENSORS_MEMBER else ZIP_STORED,
            ),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)compress|stored")


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../metadata.json",
        "/metadata.json",
        "metadata/../metadata.json",
        r"C:\metadata.json",
        r"directory\metadata.json",
    ],
)
def test_reader_rejects_traversing_or_unsafe_member_names_before_tensor_load(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    path = tmp_path / "unsafe.ckpt"
    _write_archive(
        path,
        members=[
            (unsafe_name, _metadata_bytes(), ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)unsafe|path|member name")


@pytest.mark.parametrize("missing_member", [METADATA_MEMBER, TENSORS_MEMBER])
def test_reader_rejects_a_missing_required_member_before_tensor_load(
    tmp_path: Path,
    missing_member: str,
) -> None:
    path = tmp_path / f"missing-{missing_member}.ckpt"
    members = [
        (METADATA_MEMBER, _metadata_bytes(), ZIP_STORED),
        (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
    ]
    _write_archive(path, members=[member for member in members if member[0] != missing_member])

    _assert_rejected_before_tensor_load(path, match=rf"(?i)missing.*{missing_member}")


def test_reader_rejects_an_extra_member_before_tensor_load(tmp_path: Path) -> None:
    path = tmp_path / "extra.ckpt"
    _write_archive(
        path,
        members=[
            (METADATA_MEMBER, _metadata_bytes(), ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
            ("notes.txt", b"not part of the format", ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)extra|unexpected.*notes\.txt")


def test_reader_rejects_oversized_metadata_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized-metadata.ckpt"
    metadata = _metadata_bytes(extra_json='"padding":"xxxxxxxxxxxxxxxx"')
    _write_archive(
        path,
        members=[
            (METADATA_MEMBER, metadata, ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
        ],
    )
    module = _checkpoint_module()
    monkeypatch.setattr(module, "MAX_METADATA_BYTES", len(metadata) - 1)

    _assert_rejected_before_tensor_load(path, match=r"(?i)metadata\.json.*size|too large")


def test_reader_rejects_oversized_tensor_payload_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "oversized-tensors.ckpt"
    tensor_payload = _tiny_safetensors_bytes()
    _write_archive(path)
    module = _checkpoint_module()
    monkeypatch.setattr(module, "MAX_TENSOR_BYTES", len(tensor_payload) - 1)

    _assert_rejected_before_tensor_load(path, match=r"(?i)tensors\.safetensors.*size|too large")


@pytest.mark.parametrize(
    "metadata",
    [
        b"",
        b"{",
        b"[]",
        b'"not an object"',
        b"\xff",
    ],
)
def test_reader_rejects_malformed_json_before_tensor_load(
    tmp_path: Path,
    metadata: bytes,
) -> None:
    path = tmp_path / "malformed-json.ckpt"
    _write_archive(
        path,
        members=[
            (METADATA_MEMBER, metadata, ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)metadata|json|object")


def test_reader_rejects_duplicate_json_keys_before_tensor_load(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-json-key.ckpt"
    _write_archive(
        path,
        members=[
            (METADATA_MEMBER, b'{"schema_version":2,"schema_version":2}', ZIP_STORED),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)duplicate.*schema_version")


@pytest.mark.parametrize("non_finite", ["NaN", "Infinity", "-Infinity"])
def test_reader_rejects_non_finite_json_numbers_before_tensor_load(
    tmp_path: Path,
    non_finite: str,
) -> None:
    path = tmp_path / "non-finite-json.ckpt"
    _write_archive(
        path,
        members=[
            (
                METADATA_MEMBER,
                _metadata_bytes(extra_json=f'"metric":{non_finite}'),
                ZIP_STORED,
            ),
            (TENSORS_MEMBER, _tiny_safetensors_bytes(), ZIP_STORED),
        ],
    )

    _assert_rejected_before_tensor_load(path, match=r"(?i)finite|constant|json")


class _RecordingEngine:
    def __init__(
        self,
        method: str,
        *,
        max_consumed_records: int,
        variance: float,
        visits: list[tuple[str, int, int, int, str]],
    ) -> None:
        self.method = method
        self.max_consumed_records = max_consumed_records
        self.variance = variance
        self.visits = visits

    def train_step(self, example: object, position: ExperimentPosition) -> StepResult:
        records = example if isinstance(example, tuple) else (example,)
        first_position = position.example_position - len(records) + 1
        for offset, record in enumerate(records):
            self.visits.append(
                (
                    str(record),
                    position.epoch,
                    first_position + offset,
                    position.optimizer_call_count,
                    self.method,
                )
            )
        return StepResult(
            position=position,
            language_model_loss=1.0,
            total_objective=1.0,
            regularizer_loss=0.0,
            shared_variance=self.variance,
            consumed_record_count=len(records),
            next_example_position=position.example_position + 1,
            optimizer_call_count=position.optimizer_call_count,
        )


class _FixedEvaluator:
    def evaluate(self, position: ExperimentPosition) -> EvaluationResult:
        return EvaluationResult(position, 1.0, 0.015, 0.0)


class _BoundaryStop(RuntimeError):
    pass


# covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume within an epoch
def test_observation_checkpoint_resumes_after_a_partial_eggroll_batch() -> None:
    visits: list[tuple[str, int, int, int, str]] = []
    eggroll = _RecordingEngine(
        "eggroll",
        max_consumed_records=2,
        variance=0.03,
        visits=visits,
    )
    gradient = _RecordingEngine(
        "gradient",
        max_consumed_records=1,
        variance=0.015,
        visits=visits,
    )
    saved: list[CheckpointSchedule] = []

    def stop_at_first_boundary(
        schedule: CheckpointSchedule,
        *,
        phase_boundary: bool,
        epoch_boundary: bool,
        evaluation_record: EvaluationRecord[EvaluationResult],
    ) -> tuple[Path, ...]:
        del epoch_boundary, evaluation_record
        saved.append(schedule)
        if phase_boundary:
            raise _BoundaryStop
        return ()

    with pytest.raises(_BoundaryStop):
        run_alternating._run_schedule(
            examples=["a", "b", "c", "d", "e", "f"],
            epochs=1,
            phase_steps=3,
            log_every=6,
            eggroll_engine=eggroll,
            gradient_engine=gradient,
            evaluator=_FixedEvaluator(),
            save_boundary=stop_at_first_boundary,
            output=StringIO(),
        )

    checkpoint = saved[-1]
    assert checkpoint == CheckpointSchedule(
        "gradient",
        0,
        3,
        1,
        2,
        eggroll_optimizer_calls=2,
    )
    run_alternating._run_schedule(
        examples=["a", "b", "c", "d", "e", "f"],
        epochs=1,
        phase_steps=3,
        log_every=6,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
        evaluator=_FixedEvaluator(),
        save_boundary=lambda *_args, **_kwargs: (),
        output=StringIO(),
        resume=checkpoint,
    )

    assert [(record, position) for record, _epoch, position, _call, _method in visits] == [
        ("a", 0),
        ("b", 1),
        ("c", 2),
        ("d", 3),
        ("e", 4),
        ("f", 5),
    ]
    assert [call for _record, _epoch, _position, call, method in visits if method == "eggroll"] == [
        1,
        1,
        2,
    ]


# covers: train/alternating-cycle :: Resumable experiment checkpoints :: Resume at an epoch boundary
def test_epoch_checkpoint_starts_next_epoch_with_exact_partial_window() -> None:
    visits: list[tuple[str, int, int, int, str]] = []
    eggroll = _RecordingEngine(
        "eggroll",
        max_consumed_records=2,
        variance=0.015,
        visits=visits,
    )
    gradient = _RecordingEngine(
        "gradient",
        max_consumed_records=1,
        variance=0.015,
        visits=visits,
    )
    saved: list[CheckpointSchedule] = []

    def save(
        schedule: CheckpointSchedule,
        *,
        phase_boundary: bool,
        epoch_boundary: bool,
        evaluation_record: EvaluationRecord[EvaluationResult],
    ) -> tuple[Path, ...]:
        del phase_boundary, epoch_boundary, evaluation_record
        saved.append(schedule)
        return ()

    run_alternating._run_schedule(
        examples=["a", "b", "c"],
        epochs=1,
        phase_steps=5,
        log_every=3,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
        evaluator=_FixedEvaluator(),
        save_boundary=save,
        output=StringIO(),
    )
    epoch_boundary = saved[-1]
    assert epoch_boundary.completed_phase_steps == 3
    assert epoch_boundary.phase_variance_sum == pytest.approx(0.045)
    loaded_epoch_boundary = CheckpointSchedule(
        epoch_boundary.active_phase,
        epoch_boundary.completed_phase_steps,
        epoch_boundary.global_step,
        epoch_boundary.epoch,
        -1,
        phase_variance_sum=epoch_boundary.phase_variance_sum,
        gradient_optimizer_calls=epoch_boundary.gradient_optimizer_calls,
        eggroll_optimizer_calls=epoch_boundary.eggroll_optimizer_calls,
    )

    run_alternating._run_schedule(
        examples=["a", "b", "c"],
        epochs=2,
        phase_steps=5,
        log_every=3,
        eggroll_engine=eggroll,
        gradient_engine=gradient,
        evaluator=_FixedEvaluator(),
        save_boundary=save,
        output=StringIO(),
        resume=loaded_epoch_boundary,
    )

    assert [(record, epoch, position) for record, epoch, position, _call, _method in visits] == [
        ("a", 1, 0),
        ("b", 1, 1),
        ("c", 1, 2),
        ("a", 2, 0),
        ("b", 2, 1),
        ("c", 2, 2),
    ]
    assert [call for _record, _epoch, _position, call, method in visits if method == "eggroll"] == [
        1,
        1,
        2,
        3,
        3,
        4,
    ]
