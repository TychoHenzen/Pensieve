"""Repository-wide pytest configuration."""

import os

# transformers materializes safetensors weights on a thread pool by default.
# On this Windows host that intermittently dies with an access violation in
# torch/storage.py __getitem__ while loading the Qwen backbone, killing the
# whole pytest process. HF_DEACTIVATE_ASYNC_LOAD makes the load serial
# (checked in transformers/core_model_loading.py); slower, but stable.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
