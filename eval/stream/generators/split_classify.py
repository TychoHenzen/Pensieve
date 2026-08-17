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
`source`. By default this generator fills the payload with synthetic
clusters and `source` stays `None`. Setting `data_source="mnist"` in
the config switches the feature draw to real Split-MNIST data (the
Modified National Institute of Standards and Technology set of written
digits, split into tasks of `classes_per_task` digits each): `features`
becomes 784 pixel intensities normalized to `[0, 1]`, `label` becomes
the digit string, and `source` names the dataset item (for example
`"mnist-train-12345"`). The schema does not change either way.

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


class _MnistData:
    """MNIST pixel data, indexed by digit, for both the train and test splits.

    Loading `torchvision.datasets.MNIST` is what makes `data_source="mnist"`
    an opt-in path: importing `torchvision` and touching disk only happens
    when a config actually asks for real data, so the synthetic default
    stays free of that dependency.
    """

    def __init__(self, data_dir: str) -> None:
        from torchvision.datasets import MNIST

        self.train = MNIST(root=data_dir, train=True, download=True)
        self.test = MNIST(root=data_dir, train=False, download=True)
        self._train_by_digit = self._index_by_digit(self.train.targets.tolist())
        self._test_by_digit = self._index_by_digit(self.test.targets.tolist())

    @staticmethod
    def _index_by_digit(targets: list[int]) -> dict[int, list[int]]:
        by_digit: dict[int, list[int]] = {}
        for index, digit in enumerate(targets):
            by_digit.setdefault(digit, []).append(index)
        return by_digit

    @staticmethod
    def _pixels(dataset: object, index: int) -> list[float]:
        return (dataset.data[index].flatten().float() / 255.0).tolist()  # type: ignore[attr-defined]

    def sample_train(self, source: random.Random, digit: int) -> tuple[list[float], str]:
        index = source.choice(self._train_by_digit[digit])
        return self._pixels(self.train, index), f"mnist-train-{index}"

    def sample_test(self, source: random.Random, digit: int) -> tuple[list[float], str]:
        index = source.choice(self._test_by_digit[digit])
        return self._pixels(self.test, index), f"mnist-test-{index}"


# Loading MNIST touches disk and, on first use, downloads the dataset. This
# cache keeps a `generate()` call from redoing that work on every task and
# every probe, keyed by the data directory so two configs pointing at two
# directories do not share a cache entry.
_mnist_cache: dict[str, _MnistData] = {}


def _get_mnist_data(data_dir: str) -> _MnistData:
    data = _mnist_cache.get(data_dir)
    if data is None:
        data = _MnistData(data_dir)
        _mnist_cache[data_dir] = data
    return data


class SplitClassifyGenerator:
    """Teach classification tasks in sequence, probing every task seen so far."""

    name = "split-classify"
    version = "3"

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
        data_source = params.get("data_source")

        if num_tasks < 1:
            raise ValueError(f"num_tasks must be at least 1, got {num_tasks}")
        if classes_per_task < 1:
            raise ValueError(f"classes_per_task must be at least 1, got {classes_per_task}")
        if examples_per_task < 1:
            raise ValueError(f"examples_per_task must be at least 1, got {examples_per_task}")
        if probes_per_task < 1:
            raise ValueError(f"probes_per_task must be at least 1, got {probes_per_task}")

        mnist: _MnistData | None = None
        if data_source == "mnist":
            if num_tasks * classes_per_task > 10:
                raise ValueError(
                    "num_tasks * classes_per_task must be at most 10 for MNIST, "
                    f"got {num_tasks} * {classes_per_task} = {num_tasks * classes_per_task}"
                )
            data_dir = params.get("data_dir", "./data")
            mnist = _get_mnist_data(data_dir)

        example_source = derive(seed, "examples")
        probe_source = derive(seed, "probes")

        position = 0
        for task_index in range(num_tasks):
            for _ in range(examples_per_task):
                class_index = example_source.randrange(classes_per_task)
                if mnist is not None:
                    digit = task_index * classes_per_task + class_index
                    features, source = mnist.sample_train(example_source, digit)
                    label = str(digit)
                else:
                    features = _draw_features(example_source, classes_per_task, class_index)
                    label = _class_label(task_index, class_index)
                    source = None
                yield StreamItem(
                    event=Observe(
                        position=position,
                        payload={
                            "features": features,
                            "label": label,
                            "source": source,
                        },
                    ),
                    truth=None,
                )
                position += 1

            yield StreamItem(
                event=Boundary(
                    position=position,
                    kind=BoundaryKind.TASK_TRAINED,
                    hidden_from_subject=hidden_from_subject,
                ),
                truth=None,
            )
            position += 1

            for probed_task in range(task_index + 1):
                for _ in range(probes_per_task):
                    class_index = probe_source.randrange(classes_per_task)
                    if mnist is not None:
                        digit = probed_task * classes_per_task + class_index
                        features, _source = mnist.sample_test(probe_source, digit)
                        answer: object = str(digit)
                    else:
                        features = _draw_features(probe_source, classes_per_task, class_index)
                        answer = _class_label(probed_task, class_index)
                    yield StreamItem(
                        event=Probe(
                            position=position,
                            probe_id=f"split-classify-{task_index}-{probed_task}-{position}",
                            task_id=f"task{probed_task}",
                            query=format_features(features),
                        ),
                        truth=ProbeTruth(answer=answer),
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
