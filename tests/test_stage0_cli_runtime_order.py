"""Stage 0 commands configure determinism before CUDA-sensitive parsing."""

from __future__ import annotations

import pytest

from train import run_alternating, run_eggroll, run_training


class _ParsingStopped(Exception):
    pass


@pytest.mark.parametrize(
    "module",
    [run_training, run_eggroll, run_alternating],
)
def test_deterministic_runtime_is_configured_before_argument_parser_defaults(
    monkeypatch: pytest.MonkeyPatch, module: object
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        module,
        "configure_deterministic_runtime",
        lambda: events.append("configured"),
    )

    def stop_parsing(*_args: object) -> None:
        assert events == ["configured"]
        events.append("parsed")
        raise _ParsingStopped

    monkeypatch.setattr(module, "_parse_args", stop_parsing)

    with pytest.raises(_ParsingStopped):
        module.main()

    assert events == ["configured", "parsed"]
