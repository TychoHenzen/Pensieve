"""Difficulty-costs-real-work generator: arithmetic chains, difficulty on truth.

An item labeled hard has to be harder to answer. Otherwise "compute per
input tracks difficulty" cannot be measured. So each probe is a
self-contained arithmetic chain. It starts at a number, applies a sequence
of add and subtract steps, then reports the running total modulo
`CHAIN_MODULUS`. Difficulty is the chain length, meaning the number of add
and subtract steps. A longer chain genuinely costs more arithmetic to
answer, rather than carrying a different tag. The modulus bounds the
answer to a known set of `CHAIN_MODULUS` integers. A chance rate is
therefore computable however long a chain runs.

Difficulty lives only on the `ProbeTruth` side channel. The rendered
`query` states the chain in full. It never states the chain length as a
label. A subject therefore cannot read difficulty off the text and fake
the correlation between compute spent and difficulty.

Non-probe positions between probes are `Idle` events. This generator needs
no corpus. The prose corpus belongs to `assoc`'s filler, and a chain probe
carries everything it needs inside its own `query`.
"""

from __future__ import annotations

from collections.abc import Iterator

from eval.stream.config import StreamConfig
from eval.stream.events import Idle, Probe
from eval.stream.generator import derive
from eval.stream.truth import ProbeTruth, StreamItem

# The chain runs modulo this value. The answer space is therefore always
# exactly this many integers, 0 through CHAIN_MODULUS - 1, however long a
# chain gets. That bound lets a chance rate be 1 / CHAIN_MODULUS. Without
# it the rate would be undefined over an unbounded integer range.
CHAIN_MODULUS = 1000

# Each chain step adds or subtracts an operand drawn from this closed
# range, so the rendered step is always a single-digit instruction.
_OPERAND_MIN = 1
_OPERAND_MAX = 9


def _validate(num_items: int, difficulty_levels: list[int], probe_rate: float) -> None:
    if num_items <= 0:
        raise ValueError(f"num_items must be positive, got {num_items}")
    if not difficulty_levels:
        raise ValueError("difficulty_levels must not be empty")
    for level in difficulty_levels:
        if level < 1:
            raise ValueError(f"every difficulty level must be >= 1, got {level}")
    if num_items < len(difficulty_levels):
        raise ValueError(
            f"num_items ({num_items}) must be >= the number of difficulty "
            f"levels ({len(difficulty_levels)}) so every level can appear"
        )
    if not (0 < probe_rate <= 1):
        raise ValueError(f"probe_rate must be in (0, 1], got {probe_rate}")


def _assign_difficulties(
    num_items: int, difficulty_levels: list[int], source
) -> list[int]:
    """Assign one difficulty per item so that every level appears at least once.

    The first `len(difficulty_levels)` items get one level each, shuffled,
    so every requested level is guaranteed to appear even for the smallest
    valid `num_items`. Remaining items draw a level uniformly at random.
    """
    guaranteed = list(difficulty_levels)
    source.shuffle(guaranteed)
    remaining = num_items - len(guaranteed)
    extra = [source.choice(difficulty_levels) for _ in range(remaining)]
    assignment = guaranteed + extra
    source.shuffle(assignment)
    return assignment


def _build_chain(length: int, source) -> tuple[int, list[tuple[str, int]], int]:
    """Draw a chain start and `length` (op, operand) steps, and its answer.

    Returns `(start, steps, answer)`. `answer` is the running total after
    every step, taken modulo `CHAIN_MODULUS`, matching exactly how the
    rendered `query` describes the computation.
    """
    start = source.randrange(CHAIN_MODULUS)
    steps: list[tuple[str, int]] = []
    total = start
    for _ in range(length):
        op = source.choice(("add", "subtract"))
        operand = source.randint(_OPERAND_MIN, _OPERAND_MAX)
        steps.append((op, operand))
        total = (total + operand) % CHAIN_MODULUS if op == "add" else (total - operand) % CHAIN_MODULUS
    return start, steps, total


def _render_query(start: int, steps: list[tuple[str, int]]) -> str:
    words = {"add": "Add", "subtract": "Subtract"}
    parts = [f"Start at {start}."]
    parts.extend(f"{words[op]} {operand}." for op, operand in steps)
    parts.append(f"What is the result modulo {CHAIN_MODULUS}?")
    return " ".join(parts)


class DifficultyMixGenerator:
    """Interleave arithmetic-chain probes of mixed difficulty with idle filler."""

    name = "difficulty-mix"
    version = "1"

    def chance_rate(self, config: StreamConfig) -> float:
        """Return the accuracy a random answerer reaches on this generator's probes.

        Every probe answer is an integer modulo `CHAIN_MODULUS`, so chance
        is one over that modulus regardless of chain length or `config`.
        """
        return 1.0 / CHAIN_MODULUS

    def generate(self, config: StreamConfig, seed: int) -> Iterator[StreamItem]:
        params = config.params
        num_items = params["num_items"]
        difficulty_levels = list(params["difficulty_levels"])
        probe_rate = float(params["probe_rate"])
        _validate(num_items, difficulty_levels, probe_rate)

        total_length = max(num_items, round(num_items / probe_rate))

        difficulty_source = derive(seed, "difficulty")
        chain_source = derive(seed, "chain")
        order_source = derive(seed, "order")
        idle_source = derive(seed, "idle")

        difficulties = _assign_difficulties(num_items, difficulty_levels, difficulty_source)
        probe_positions = sorted(order_source.sample(range(total_length), num_items))
        difficulty_by_position = dict(zip(probe_positions, difficulties, strict=True))

        for position in range(total_length):
            if position not in difficulty_by_position:
                yield StreamItem(
                    event=Idle(position=position, budget=idle_source.randint(1, 5)),
                    truth=None,
                )
                continue
            difficulty = difficulty_by_position[position]
            start, steps, answer = _build_chain(difficulty, chain_source)
            yield StreamItem(
                event=Probe(
                    position=position,
                    probe_id=f"difficulty-mix-{position}",
                    task_id="difficulty-mix",
                    query=_render_query(start, steps),
                ),
                truth=ProbeTruth(answer=answer, difficulty=difficulty),
            )
