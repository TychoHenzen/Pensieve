# Tech stack

- Python 3.13 or newer. Package metadata uses setuptools in `pyproject.toml`.
- Local environment: `.venv\Scripts\python.exe` on Windows PowerShell.
- Test and quality tools: pytest and Ruff. Ruff targets Python 3.13 and 120-character lines.
- Optional dependency groups: `dev` for pytest/numpy/scipy/Ruff; `corpus` for zstandard; `baselines` for torch/torchvision; `stage0` for datasets/transformers/sentence-transformers/torch.
- Model work depends on PyTorch, Transformers, Sentence Transformers, and Qwen-family checkpoints. Numerical and dependency drift can invalidate reproducibility evidence.
- Serena uses the Python LSP backend with Pyright on Windows.