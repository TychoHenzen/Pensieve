"""Ordering contract for the Stage 0 subject (LatentCoreSubject).

Pins the core/latent-loop "Subject implements the v1 subject protocol"
scenarios that the integration tests exercise only through a real backbone:
observe encodes before it runs the latent loop, and answer runs the latent
loop before it decodes. The encoder, latent loop, and decoder are replaced
with recording fakes, so these tests run fast and never load a model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from eval.stream.events import Observe, Probe
from eval.subjects.latent_core import LatentCoreSubject

SLOT_COUNT = 4

PROTOCOL_METHODS = ("observe", "answer", "idle", "snapshot", "restore", "cost")


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def record(self, name: str) -> None:
        self.calls.append(name)


def _build_subject(
    encoder: MagicMock,
    loop: MagicMock,
    decoder: MagicMock,
) -> LatentCoreSubject:
    with (
        patch(
            "eval.subjects.latent_core.load_frozen_qwen_backbone",
            return_value=MagicMock(),
        ),
        patch("eval.subjects.latent_core.SlotEncoder", return_value=encoder),
        patch("eval.subjects.latent_core.LatentLoop", return_value=loop),
        patch("eval.subjects.latent_core.SlotDecoder", return_value=decoder),
    ):
        return LatentCoreSubject(slot_count=SLOT_COUNT, num_steps=2, device="cpu")


# covers: core/latent-loop::Subject implements the v1 subject protocol::all protocol methods present
def test_subject_implements_all_six_protocol_methods() -> None:
    for name in PROTOCOL_METHODS:
        method = getattr(LatentCoreSubject, name, None)
        assert method is not None, f"LatentCoreSubject is missing {name!r}"
        assert callable(method), f"LatentCoreSubject.{name} is not callable"


# covers: core/latent-loop::Subject implements the v1 subject protocol::observe runs encoder then latent loop
def test_observe_runs_encoder_then_latent_loop() -> None:
    recorder = _Recorder()
    encoder = MagicMock()
    loop = MagicMock()

    def encode_to_workspace(*args: object, **kwargs: object) -> None:
        recorder.record("encode")

    def run(*args: object, **kwargs: object) -> None:
        recorder.record("loop.run")

    encoder.encode_to_workspace.side_effect = encode_to_workspace
    loop.run.side_effect = run

    subject = _build_subject(encoder=encoder, loop=loop, decoder=MagicMock())
    event = Observe(position=0, payload={"text": "the sky is blue"})

    subject.observe(event)

    assert recorder.calls == ["encode", "loop.run"]
    assert encoder.encode_to_workspace.call_args.args[0] == "the sky is blue"
    assert encoder.encode_to_workspace.call_args.args[1] is subject.workspace
    assert loop.run.call_args.args[0] is subject.workspace


# covers: core/latent-loop::Subject implements the v1 subject protocol::answer runs latent loop then decoder
def test_answer_runs_latent_loop_then_decoder() -> None:
    recorder = _Recorder()
    loop = MagicMock()
    decoder = MagicMock()

    def run(*args: object, **kwargs: object) -> None:
        recorder.record("loop.run")

    def decode(*args: object, **kwargs: object) -> str:
        recorder.record("decode")
        return "decoded answer"

    loop.run.side_effect = run
    decoder.decode.side_effect = decode

    subject = _build_subject(encoder=MagicMock(), loop=loop, decoder=decoder)
    probe = Probe(position=0, probe_id="p0", task_id="t0", query="what color?")

    answer = subject.answer(probe)

    assert recorder.calls == ["loop.run", "decode"]
    assert loop.run.call_args.args[0] is subject.workspace
    assert decoder.decode.call_args.args[0] is subject.workspace
    assert answer == "decoded answer"
