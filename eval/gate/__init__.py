"""Stage 0 gate scripts: token-CoT baseline, latent evaluation, ablations, report.

Each script under this package is a standalone CLI entry point
(`if __name__ == "__main__": main()`) that writes its results as JSON
under `gate_results/`. `gate_report.py` collects those JSON files and
renders the pass/fail determination for the Stage 0 gate.
"""

from __future__ import annotations
