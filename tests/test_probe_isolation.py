"""Probe isolation: a snapshot/restore cycle must leave outputs unchanged.

`isolated_answer` snapshots the subject, answers a probe, then restores.
This test checks a stronger property than that alone: that restoring after
seeing *extra* noise input between the snapshot and the restore produces
the exact same downstream answer as a subject that never saw the noise at
all. That is the isolation guarantee the harness depends on to score a
probe without letting it teach the subject anything.

Real Pythia-160M and MiniLM models load here, on purpose: this is the
end-to-end integration check, not a unit test standing in for one.
"""

from __future__ import annotations

from eval.stream.events import Observe, Probe
from eval.subjects.latent_core import LatentCoreSubject

SLOT_COUNT = 4
NUM_STEPS = 1

_INITIAL_OBSERVE = Observe(position=0, payload={"text": "The sky is blue."})
_NOISE_OBSERVE = Observe(
    position=1, payload={"text": "Purple elephants juggle flaming kettles."}
)
_PROBE = Probe(
    position=2,
    probe_id="p0",
    task_id="t0",
    query="What color is the sky?",
)


def _make_subject() -> LatentCoreSubject:
    return LatentCoreSubject(slot_count=SLOT_COUNT, num_steps=NUM_STEPS, device="cpu")


def test_snapshot_restore_isolates_noise_from_answer() -> None:
    # A single subject instance is used throughout. Comparing two separate
    # `LatentCoreSubject` instances would compare two independently
    # randomly initialized encoder/decoder heads, which differ regardless
    # of isolation. The isolation property under test is about one
    # subject's state over time, not about matching two instances.
    subject = _make_subject()
    subject.observe(_INITIAL_OBSERVE)
    snapshot = subject.snapshot()

    # Reference: answer the probe from the snapshotted state, with no
    # noise ever having been fed to the subject.
    reference_answer = subject.answer(_PROBE)

    # Restore to the pre-answer snapshot, feed noise, then restore again
    # before answering. If the restore is a true isolation boundary, this
    # answer must be identical to the reference above.
    subject.restore(snapshot)
    subject.observe(_NOISE_OBSERVE)
    subject.restore(snapshot)
    restored_answer = subject.answer(_PROBE)

    assert restored_answer == reference_answer
