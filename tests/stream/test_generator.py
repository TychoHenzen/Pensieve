from collections.abc import Iterator

from eval.stream.config import StreamConfig
from eval.stream.events import Observe
from eval.stream.generator import StreamGenerator, derive
from eval.stream.truth import StreamItem


def _draws(seed, name, count=5):
    source = derive(seed, name)
    return [source.random() for _ in range(count)]


# covers: eval/generator::Independent named random sources::same seed and name repeat
def test_derive_same_seed_and_name_repeats_the_sequence():
    assert _draws(0, "keys") == _draws(0, "keys")


# covers: eval/generator::Independent named random sources::different names diverge
def test_derive_different_names_diverge():
    assert _draws(0, "keys") != _draws(0, "order")


def test_derive_different_seeds_diverge():
    assert _draws(0, "keys") != _draws(1, "keys")


def test_derive_sources_are_independent_of_draw_order():
    keys_first = derive(0, "keys")
    order_first = derive(0, "order")
    keys_first.random()  # draw from "keys" before touching "order"
    untouched_order = derive(0, "order")
    assert order_first.random() == untouched_order.random()


# covers: eval/generator::Independent named random sources::known seed produces a known first draw
def test_derive_is_not_seeded_by_process_hash_randomization():
    # A known value derived from sha256(b"0:keys"), independent of PYTHONHASHSEED.
    source = derive(0, "keys")
    assert source.random() == 0.9233467507638282


class _FakeGenerator:
    """Minimal Protocol-satisfying generator for structural checks."""

    name = "fake"
    version = "1"

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        source = derive(seed, "payload")
        for position in range(3):
            yield StreamItem(
                event=Observe(position=position, payload=source.random()),
                truth=None,
            )


# covers: eval/generator::StreamGenerator protocol shape::protocol conformance
def test_fake_generator_satisfies_protocol_statically():
    generator: StreamGenerator = _FakeGenerator()
    assert isinstance(generator, StreamGenerator)
    assert generator.name == "fake"


def test_fake_generator_yields_stream_items():
    generator = _FakeGenerator()
    items = list(generator.generate(config=None, seed=0))
    assert len(items) == 3
    assert all(isinstance(item, StreamItem) for item in items)
    assert [item.event.position for item in items] == [0, 1, 2]
