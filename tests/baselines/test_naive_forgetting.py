"""Catastrophic-forgetting test for the naive sequential baseline.

Drives `NaiveBaseline` through the spec's GIVEN - a 5-task synthetic
`split-classify` stream - and checks the spec's THEN: accuracy on the first
task is high right after it is trained, then drops sharply once the later
four tasks overwrite the model. A small MLP (one hidden layer, 32 units)
with 200 training iterations per task keeps the test under a few seconds
while leaving the qualitative forgetting behavior intact.
"""

from __future__ import annotations

import torch

from eval.baselines.naive import NaiveBaseline
from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.stream.truth import StreamItem

_NUM_TASKS = 5
_CLASSES_PER_TASK = 2
_EXAMPLES_PER_TASK = 20
_PROBES_PER_TASK = 10


def _stream() -> list[StreamItem]:
    config = StreamConfig(
        generator="split-classify",
        params={
            "num_tasks": _NUM_TASKS,
            "classes_per_task": _CLASSES_PER_TASK,
            "examples_per_task": _EXAMPLES_PER_TASK,
            "probes_per_task": _PROBES_PER_TASK,
        },
    )
    return list(SplitClassifyGenerator().generate(config, seed=0))


def _task0_accuracy_by_block(items: list[StreamItem]) -> dict[int, float]:
    """Run the naive baseline, returning task-0 accuracy per trained-task block.

    Block ``b`` is the probe batch emitted right after task ``b`` has been
    trained, so block 0 is "early" and block ``_NUM_TASKS - 1`` is "late".
    """
    torch.manual_seed(0)
    baseline = NaiveBaseline(
        input_dim=_CLASSES_PER_TASK,
        output_dim=_NUM_TASKS * _CLASSES_PER_TASK,
        hidden_layers=1,
        hidden_units=32,
        train_iterations=200,
    )

    block = -1
    correct: dict[int, int] = {}
    total: dict[int, int] = {}
    for item in items:
        event = item.event
        if isinstance(event, Boundary):
            if event.kind == BoundaryKind.TASK_TRAINED:
                block += 1
            baseline.observe(event)
        elif isinstance(event, Observe):
            baseline.observe(event)
        elif isinstance(event, Probe):
            if event.task_id == "task0":
                total[block] = total.get(block, 0) + 1
                truth = item.truth
                if truth is not None and baseline.answer(event) == truth.answer:
                    correct[block] = correct.get(block, 0) + 1

    return {b: correct.get(b, 0) / total[b] for b in total}


# covers: eval/baselines::Naive sequential baseline::catastrophic forgetting visible
def test_naive_forgets_early_task_after_later_tasks() -> None:
    accuracy = _task0_accuracy_by_block(_stream())

    early = accuracy[0]
    late = accuracy[_NUM_TASKS - 1]

    assert early > 0.8, f"task-0 accuracy right after training should be high, got {early}"
    assert late < 0.5, f"task-0 accuracy after all tasks should collapse, got {late}"
    assert early - late > 0.5, f"forgetting should be sharp, got {early} -> {late}"
