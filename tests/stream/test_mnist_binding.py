import re

from eval.stream.config import StreamConfig
from eval.stream.events import Observe
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.stream.hashing import stream_hash

MNIST_PARAMS = {
    "num_tasks": 2,
    "classes_per_task": 2,
    "examples_per_task": 4,
    "probes_per_task": 2,
}


def _config(**overrides):
    params = dict(MNIST_PARAMS)
    params.update(overrides)
    return StreamConfig(generator="split-classify", params=params)


def _observes(config, seed=0):
    generator = SplitClassifyGenerator()
    items = list(generator.generate(config=config, seed=seed))
    return [item.event for item in items if isinstance(item.event, Observe)]


# covers: eval/generators::MNIST data binding::synthetic mode unchanged without data_source
def test_synthetic_mode_unchanged_without_data_source():
    observes = _observes(_config())
    assert observes
    for event in observes:
        assert event.payload["source"] is None
        assert len(event.payload["features"]) == MNIST_PARAMS["classes_per_task"]


# covers: eval/generators::MNIST data binding::mnist features are 784 elements with a non-none source
def test_mnist_features_are_784_elements_with_source():
    observes = _observes(_config(data_source="mnist", data_dir="./data"))
    assert observes
    for event in observes:
        assert len(event.payload["features"]) == 784
        assert event.payload["source"] is not None
        assert re.fullmatch(r"mnist-train-\d+", event.payload["source"])


# covers: eval/generators::MNIST data binding::mnist labels are digit strings
def test_mnist_labels_are_digit_strings():
    observes = _observes(_config(data_source="mnist", data_dir="./data"))
    assert observes
    for event in observes:
        assert re.fullmatch(r"\d+", event.payload["label"])


# covers: eval/generators::MNIST data binding::stream hash differs between synthetic and mnist configs
def test_stream_hash_differs_between_synthetic_and_mnist():
    synthetic_config = _config()
    mnist_config = _config(data_source="mnist", data_dir="./data")

    synthetic_hash = stream_hash(
        synthetic_config,
        seed=0,
        generator_version="3",
        render_version="1",
        corpus_id="corpus",
    )
    mnist_hash = stream_hash(
        mnist_config,
        seed=0,
        generator_version="3",
        render_version="1",
        corpus_id="corpus",
    )

    assert synthetic_hash != mnist_hash


# covers: eval/generators::MNIST data binding::mnist feature values are in [0, 1]
def test_mnist_feature_values_are_in_valid_range():
    observes = _observes(_config(data_source="mnist", data_dir="./data"))
    assert observes
    for event in observes:
        for value in event.payload["features"]:
            assert 0.0 <= value <= 1.0
