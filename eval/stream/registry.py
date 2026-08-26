"""Generator name resolution, so a config alone is enough to build a stream.

A run record stores a `StreamConfig` and nothing else. Rebuilding the run
therefore means turning `config.generator`, a plain string, back into the
generator class that produced it. `REGISTRY` holds that mapping and `build`
does the lookup, so a run record never needs to carry a class reference or
an import path.
"""

from __future__ import annotations

from collections.abc import Iterator

from eval.stream.config import StreamConfig
from eval.stream.generator import StreamGenerator
from eval.stream.generators.assoc import AssocGenerator
from eval.stream.generators.difficulty_mix import DifficultyMixGenerator
from eval.stream.generators.asdiv_a import AsdivGenerator
from eval.stream.generators.gsm8k import GSM8KGenerator
from eval.stream.generators.split_classify import SplitClassifyGenerator
from eval.stream.truth import StreamItem

REGISTRY: dict[str, type[StreamGenerator]] = {
    AssocGenerator.name: AssocGenerator,
    SplitClassifyGenerator.name: SplitClassifyGenerator,
    DifficultyMixGenerator.name: DifficultyMixGenerator,
    AsdivGenerator.name: AsdivGenerator,
    GSM8KGenerator.name: GSM8KGenerator,
}


def build(config: StreamConfig, seed: int) -> Iterator[StreamItem]:
    """Resolve `config.generator` in `REGISTRY` and generate its stream.

    Raises `ValueError` naming the known generators when `config.generator`
    is not registered.
    """
    generator_class = REGISTRY.get(config.generator)
    if generator_class is None:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"unknown generator {config.generator!r}; known generators: {known}"
        )
    return generator_class().generate(config=config, seed=seed)
