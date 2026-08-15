import re

import pytest

from eval.stream.config import StreamConfig
from eval.stream.events import Boundary, BoundaryKind, Observe, Probe
from eval.stream.generator import StreamGenerator
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.stream.render import render_event


def _config(**overrides):
    params = {
        "num_tasks": 3,
        "classes_per_task": 2,
        "examples_per_task": 4,
        "probes_per_task": 2,
    }
    params.update(overrides)
    return StreamConfig(generator="split-classify", params=params)


def _items(config=None, seed=0):
    generator = SplitClassifyGenerator()
    return list(generator.generate(config=config or _config(), seed=seed))


# covers: eval/generators/split-classify::Generator identity and version::version is pinned
def test_generator_satisfies_stream_generator_protocol():
    generator = SplitClassifyGenerator()
    assert isinstance(generator, StreamGenerator)
    assert generator.name == "split-classify"
    assert generator.version == "2"


# covers: eval/generators/split-classify::Positions are contiguous from zero::contiguous positions
def test_positions_are_consecutive_from_zero_with_no_gaps_or_repeats():
    items = _items()
    positions = [item.event.position for item in items]
    assert positions == list(range(len(positions)))


# covers: eval/generators/split-classify::Observe payload has fixed shape::payload shape
def test_observe_payload_holds_features_label_and_source():
    items = _items()
    observes = [item.event for item in items if isinstance(item.event, Observe)]
    assert observes
    for event in observes:
        assert set(event.payload.keys()) == {"features", "label", "source"}
        assert isinstance(event.payload["features"], list)
        assert event.payload["features"]
        assert isinstance(event.payload["label"], str)
        assert event.payload["source"] is None


# covers: eval/generators/split-classify::Sequential task blocks with examples then probes::probes after task j cover every task up to j
def test_probes_after_task_j_cover_every_task_up_to_j():
    config = _config(num_tasks=4, classes_per_task=3, examples_per_task=2, probes_per_task=2)
    items = _items(config)

    # Walk the stream task block by task block: each block is the
    # examples for task j followed by the probes that immediately
    # follow it, ending at the next Boundary or the stream's end.
    task_ids_by_block: list[set[str]] = []
    current_probe_task_ids: set[str] = set()
    for item in items:
        if isinstance(item.event, Probe):
            current_probe_task_ids.add(item.event.task_id)
        elif isinstance(item.event, Boundary):
            task_ids_by_block.append(current_probe_task_ids)
            current_probe_task_ids = set()
    task_ids_by_block.append(current_probe_task_ids)

    assert len(task_ids_by_block) == 4
    for task_index, probe_task_ids in enumerate(task_ids_by_block):
        expected = {f"task{i}" for i in range(task_index + 1)}
        assert probe_task_ids == expected


# covers: eval/generators/split-classify::Sequential task blocks with examples then probes::probe count per checkpoint
def test_probes_per_task_count_matches_config_at_each_checkpoint():
    config = _config(num_tasks=3, classes_per_task=2, examples_per_task=2, probes_per_task=3)
    items = _items(config)

    counts_by_block: list[dict[str, int]] = []
    current: dict[str, int] = {}
    for item in items:
        if isinstance(item.event, Probe):
            current[item.event.task_id] = current.get(item.event.task_id, 0) + 1
        elif isinstance(item.event, Boundary):
            counts_by_block.append(current)
            current = {}
    counts_by_block.append(current)

    for task_index, counts in enumerate(counts_by_block):
        for i in range(task_index + 1):
            assert counts[f"task{i}"] == 3


# covers: eval/generators/split-classify::Boundaries between tasks but not after the last::boundaries land between tasks
def test_boundaries_land_between_tasks():
    config = _config(num_tasks=4, classes_per_task=2, examples_per_task=3, probes_per_task=1)
    items = _items(config)

    boundaries = [item for item in items if isinstance(item.event, Boundary)]
    assert len(boundaries) == 3  # one fewer than num_tasks

    for boundary_item in boundaries:
        assert boundary_item.event.kind == BoundaryKind.TASK_SWITCH
        boundary_position = boundary_item.event.position

        before = [
            item.event for item in items
            if item.event.position < boundary_position
        ]
        after = [
            item.event for item in items
            if item.event.position > boundary_position
        ]
        assert before
        assert after
        # Immediately before a boundary sits a probe (the end of a
        # task's checkpoint); immediately after sits a fresh task's
        # first Observe.
        assert isinstance(before[-1], Probe)
        assert isinstance(after[0], Observe)


# covers: eval/generators/split-classify::Boundaries between tasks but not after the last::no boundary after the final task
def test_no_boundary_after_the_final_task():
    items = _items(_config(num_tasks=2))
    assert items
    assert not isinstance(items[-1].event, Boundary)


# covers: eval/generators/split-classify::Boundary hidden_from_subject reflects config::hidden boundaries
def test_hidden_from_subject_is_configurable():
    hidden = _items(_config(hidden_from_subject=True))
    shown = _items(_config(hidden_from_subject=False))

    hidden_boundaries = [item.event for item in hidden if isinstance(item.event, Boundary)]
    shown_boundaries = [item.event for item in shown if isinstance(item.event, Boundary)]

    assert hidden_boundaries
    assert all(boundary.hidden_from_subject for boundary in hidden_boundaries)
    assert all(not boundary.hidden_from_subject for boundary in shown_boundaries)


# covers: eval/generators/split-classify::Probe truth answer is a valid label for the probed task::answer matches task
def test_probe_truth_answer_is_a_valid_label_for_its_task():
    config = _config(num_tasks=3, classes_per_task=2)
    items = _items(config)
    for item in items:
        if not isinstance(item.event, Probe):
            continue
        assert item.truth is not None
        assert item.truth.answer.startswith(item.event.task_id + "-class")


# covers: eval/generators/split-classify::Deterministic in (config, seed) and diverges on different seeds::replay determinism
def test_same_seed_gives_identical_sequence():
    first = _items()
    second = _items()
    assert [item.event for item in first] == [item.event for item in second]
    assert [item.truth for item in first] == [item.truth for item in second]


# covers: eval/generators/split-classify::Deterministic in (config, seed) and diverges on different seeds::replay determinism
def test_different_seed_gives_different_sequence():
    first = _items(seed=0)
    second = _items(seed=1)
    assert [item.event for item in first] != [item.event for item in second]


# covers: eval/generators/split-classify::Validates all four required params as >= 1::zero tasks rejected
def test_invalid_params_raise_value_error():
    with pytest.raises(ValueError):
        _items(_config(num_tasks=0))
    with pytest.raises(ValueError):
        _items(_config(classes_per_task=0))
    with pytest.raises(ValueError):
        _items(_config(examples_per_task=0))
    with pytest.raises(ValueError):
        _items(_config(probes_per_task=0))


# covers: eval/generators/split-classify::chance_rate equals 1/classes_per_task, computed from that single param alone::chance rate with minimal config
def test_chance_rate_with_minimal_config():
    config = StreamConfig(generator="split-classify", params={"classes_per_task": 5})
    generator = SplitClassifyGenerator()
    assert generator.chance_rate(config) == pytest.approx(0.2)


# covers: eval/generators/split-classify::Class clusters stay separable::cluster geometry
def test_cluster_centers_are_separable():
    from eval.stream.generators.split_classify import (
        FEATURE_NOISE_STD,
        _class_center,
    )
    import math

    num_classes = 4
    centers = [_class_center(num_classes, i) for i in range(num_classes)]
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(centers[i], centers[j])))
            assert dist > FEATURE_NOISE_STD, (
                f"centers {i} and {j} are {dist:.2f} apart, "
                f"less than noise std {FEATURE_NOISE_STD}"
            )


# covers: eval/generators/split-classify::Probe query and example use the same feature notation::feature precision alignment
def test_probe_query_and_example_state_features_in_the_same_wording():
    """A probe is compared against taught examples, so both must read alike.

    The query used to interpolate a raw Python float list, printing 17
    digits of representation noise, while the example rendered two
    decimals. A subject would then have to match two different notations
    for the same vector.
    """
    items = _items()
    probe = next(item.event for item in items if isinstance(item.event, Probe))
    example = next(item.event for item in items if isinstance(item.event, Observe))

    assert probe.query.startswith("features=(")
    assert render_event(example).startswith(f"[example] {probe.query[:10]}")
    assert re.search(r"\d\.\d{3,}", probe.query) is None
    assert re.search(r"\d\.\d{3,}", render_event(example)) is None
