from __future__ import annotations

import torch

from train.vicreg import post_loop_slot_variance


def test_given_unbatched_slots_when_measuring_post_loop_variance_then_uses_population_variance() -> None:
    slots = torch.tensor([[0.0, 0.0], [2.0, 4.0]])

    variance = post_loop_slot_variance(slots)

    assert torch.isclose(variance, torch.tensor(2.5))


def test_given_batched_slots_when_measuring_post_loop_variance_then_averages_batches_and_dimensions() -> None:
    slots = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 4.0]],
            [[0.0, 0.0], [4.0, 2.0]],
        ]
    )

    variance = post_loop_slot_variance(slots)

    assert torch.isclose(variance, torch.tensor(2.5))
