from __future__ import annotations

from eval.stream.events import Probe
from eval.subject import Subject


def isolated_answer(subject: Subject, probe: Probe) -> str:
    state = subject.snapshot()
    result = subject.answer(probe)
    subject.restore(state)
    return result
