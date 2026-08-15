"""Split-classify generator: tasks in sequence, probes on every task seen so far.

Presents `num_tasks` classification tasks one after another. Each task
teaches `examples_per_task` synthetic, separable feature vectors. Those
vectors come from `classes_per_task` class clusters. After a task's
examples, the stream probes every task seen so far, including the one
just taught. It sends `probes_per_task` probes per task. One run can
therefore fill a retention matrix indexed by the task probed and the task
just finished. `Boundary(TASK_SWITCH)` marks the seam between one task's
block and the next. A config sets `hidden_from_subject` on that boundary.

The `Observe` payload holds `features`, `label`, and an optional
`source`. `source` stays `None` here, because this generator fills the
payload with synthetic clusters. The real data would be Split-MNIST (the
Modified National Institute of Standards and Technology set of written
digits, split into tasks of two classes each). Binding it later needs a
different
`source` and a different feature draw. It needs no schema change.

Every random draw comes from `eval.stream.generator.derive(seed, name)`,
so two runs at the same seed produce the identical stream.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generator import derive
from eval.stream.render import format_features
from eval.stream.truth import ProbeTruth, StreamItem

# Distance between a class cluster's center and the origin, and the
# spread of the noise around that center. Both are fixed rather than
# configurable, because nothing in the plan asks the separation to vary.
# A wide margin between the two keeps the classes cleanly separable at
# every classes_per_task this generator is asked to build.
FEATURE_SEPARATION = 5.0
FEATURE_NOISE_STD = 0.5


def _class_label(task_index: int, class_index: int) -> str:
    return f"task{task_index}-class{class_index}"


def _class_center(classes_per_task: int, class_index: int) -> list[float]:
    center = [0.0] * classes_per_task
    center[class_index] = FEATURE_SEPARATION
    return center


def _draw_features(source: random.Random, classes_per_task: int, class_index: int) -> list[float]:
    center = _class_center(classes_per_task, class_index)
    return [value + source.gauss(0.0, FEATURE_NOISE_STD) for value in center]


class SplitClassifyGenerator:
    """Teach classification tasks in sequence, probing every task seen so far."""

    name = "split-classify"
    version = "2"

    def chance_rate(self, config: StreamConfig) -> float:
        """Return the accuracy a random answerer reaches on this generator's probes.

        Every probe answer is one of `classes_per_task` labels, so chance
        is one over that count.
        """
        classes_per_task = config.params["classes_per_task"]
        return 1.0 / classes_per_task

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        params = config.params
        num_tasks = params["num_tasks"]
        classes_per_task = params["classes_per_task"]
        examples_per_task = params["examples_per_task"]
        probes_per_task = params["probes_per_task"]
        hidden_from_subject = bool(params.get("hidden_from_subject", False))

        if num_tasks < 1:
            raise ValueError(f"num_tasks must be at least 1, got {num_tasks}")
        if classes_per_task < 1:
            raise ValueError(f"classes_per_task must be at least 1, got {classes_per_task}")
        if examples_per_task < 1:
            raise ValueError(f"examples_per_task must be at least 1, got {examples_per_task}")
        if probes_per_task < 1:
            raise ValueError(f"probes_per_task must be at least 1, got {probes_per_task}")

        example_source = derive(seed, "examples")
        probe_source = derive(seed, "probes")

        position = 0
        for task_index in range(num_tasks):
            for _ in range(examples_per_task):
                class_index = example_source.randrange(classes_per_task)
                features = _draw_features(example_source, classes_per_task, class_index)
                yield StreamItem(
                    event=Observe(
                        position=position,
                        payload={
                            "features": features,
                            "label": _class_label(task_index, class_index),
                            "source": None,
                        },
                    ),
                    truth=None,
                )
                position += 1

            for probed_task in range(task_index + 1):
                for _ in range(probes_per_task):
                    class_index = probe_source.randrange(classes_per_task)
                    features = _draw_features(probe_source, classes_per_task, class_index)
                    yield StreamItem(
                        event=Probe(
                            position=position,
                            probe_id=f"split-classify-{task_index}-{probed_task}-{position}",
                            task_id=f"task{probed_task}",
                            query=format_features(features),
                        ),
                        truth=ProbeTruth(answer=_class_label(probed_task, class_index)),
                    )
                    position += 1

            if task_index < num_tasks - 1:
                yield StreamItem(
                    event=Boundary(
                        position=position,
                        kind=BoundaryKind.TASK_SWITCH,
                        hidden_from_subject=hidden_from_subject,
                    ),
                    truth=None,
                )
                position += 1
