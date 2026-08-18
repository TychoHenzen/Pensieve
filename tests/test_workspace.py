import pytest
import torch

from workspace.concept_slots import ABLATION_SLOT_COUNTS, SLOT_DIM, Workspace


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
