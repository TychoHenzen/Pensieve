import pytest
import torch

from workspace.concept_slots import ABLATION_SLOT_COUNTS, SLOT_DIM, Workspace

HEALTHY_VARIANCE_THRESHOLD = 0.1


@pytest.mark.parametrize("slot_count", ABLATION_SLOT_COUNTS)
def test_snapshot_restore_round_trip(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)

    original_values = torch.randn(slot_count, SLOT_DIM)
    ws.write_slots(original_values)

    snapshot = ws.snapshot()

    overwrite_values = torch.randn(slot_count, SLOT_DIM)
    ws.write_slots(overwrite_values)
    assert not torch.equal(ws.read_slots(), snapshot)

    ws.restore(snapshot)

    assert torch.equal(ws.read_slots(), snapshot)
    assert torch.equal(ws.read_slots(), original_values)


@pytest.mark.parametrize("slot_count", ABLATION_SLOT_COUNTS)
def test_snapshot_is_independent_copy(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)
    ws.write_slots(torch.randn(slot_count, SLOT_DIM))

    snapshot = ws.snapshot()
    snapshot_before_mutation = snapshot.clone()

    ws.write_slots(torch.randn(slot_count, SLOT_DIM))

    assert torch.equal(snapshot, snapshot_before_mutation)


@pytest.mark.parametrize("slot_count", ABLATION_SLOT_COUNTS)
def test_restore_is_independent_copy(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)
    ws.write_slots(torch.randn(slot_count, SLOT_DIM))

    state = ws.snapshot()
    ws.restore(state)

    state.add_(1.0)

    assert not torch.equal(ws.read_slots(), state)


@pytest.mark.parametrize("slot_count", [c for c in ABLATION_SLOT_COUNTS if c >= 2])
def test_collapse_stats_high_variance_for_diverse_slots(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)
    ws.write_slots(torch.randn(slot_count, SLOT_DIM))

    stats = ws.collapse_stats()

    assert stats["variance"] > HEALTHY_VARIANCE_THRESHOLD


@pytest.mark.parametrize("slot_count", [c for c in ABLATION_SLOT_COUNTS if c >= 2])
def test_collapse_stats_near_zero_variance_for_collapsed_slots(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)
    single_vector = torch.randn(1, SLOT_DIM).expand(slot_count, SLOT_DIM).clone()
    ws.write_slots(single_vector)

    stats = ws.collapse_stats()

    assert stats["variance"] == pytest.approx(0.0, abs=1e-6)
    assert stats["covariance"] == pytest.approx(0.0, abs=1e-6)


def test_collapse_stats_single_slot_is_zero() -> None:
    ws = Workspace(slot_count=1)
    ws.write_slots(torch.randn(1, SLOT_DIM))

    stats = ws.collapse_stats()

    assert stats == {"variance": 0.0, "covariance": 0.0}


@pytest.mark.parametrize("slot_count", [c for c in ABLATION_SLOT_COUNTS if c >= 2])
def test_instrumentation_fields_reports_diverse_variance(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)
    ws.write_slots(torch.randn(slot_count, SLOT_DIM))

    fields = ws.instrumentation_fields()

    assert fields.memory_write_magnitude is not None
    assert fields.memory_write_magnitude > HEALTHY_VARIANCE_THRESHOLD


@pytest.mark.parametrize("slot_count", [c for c in ABLATION_SLOT_COUNTS if c >= 2])
def test_instrumentation_fields_reports_collapsed_variance(slot_count: int) -> None:
    ws = Workspace(slot_count=slot_count)
    single_vector = torch.randn(1, SLOT_DIM).expand(slot_count, SLOT_DIM).clone()
    ws.write_slots(single_vector)

    fields = ws.instrumentation_fields()

    assert fields.memory_write_magnitude == pytest.approx(0.0, abs=1e-6)
