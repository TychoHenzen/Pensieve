from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from eval.stage0_identity import load_minilm_model
from workspace.concept_slots import SLOT_DIM, Workspace

MINILM_DIM = 384
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_ATTENTION_HEADS = 8
MINILM_TOKEN_LIMIT = 256


def _load_sentence_transformer(
    repository: str,
    *,
    revision: str,
    trust_remote_code: bool,
    use_safetensors: bool,
) -> Any:
    return SentenceTransformer(
        repository,
        revision=revision,
        trust_remote_code=trust_remote_code,
        model_kwargs={"use_safetensors": use_safetensors},
    )


class SlotEncoder(nn.Module):
    """Encodes text into workspace slot vectors.

    A frozen pretrained sentence encoder (all-MiniLM-L6-v2) produces
    token-level embeddings. A learned linear layer projects those
    embeddings from 384 to 896 dimensions. Each output slot is a
    learned query vector plus the mean projected embedding, giving
    inter-slot diversity from the queries and input dependence from
    the mean.
    """

    def __init__(
        self,
        slot_count: int = 16,
        device: str = "cpu",
        sentence_model: Any | None = None,
        manifest_verifier: Callable[[str, str, Mapping[str, Mapping[str, str]]], None] | None = None,
    ) -> None:
        super().__init__()
        self.slot_count = slot_count
        self.device = device

        self._sentence_model = (
            load_minilm_model(
                model_loader=_load_sentence_transformer,
                manifest_verifier=manifest_verifier,
            )
            if sentence_model is None
            else sentence_model
        )
        self._sentence_model.to(device)
        for param in self._sentence_model.parameters():
            param.requires_grad = False
        self._sentence_model.eval()

        self.projection = nn.Linear(MINILM_DIM, SLOT_DIM)
        self.slot_queries = nn.Parameter(torch.randn(slot_count, SLOT_DIM))
        self.attn_log_temp = nn.Parameter(torch.tensor(0.0))

        self.to(device)

    def _token_embeddings(self, text: str) -> torch.Tensor:
        """Frozen token-level embeddings for `text`, shape (1, tokens, 384)."""
        transformer = self._sentence_model[0]
        tokenizer = transformer.tokenizer
        features = tokenizer([text], return_tensors="pt", padding=False, truncation=False)
        input_ids = features.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("MiniLM tokenizer must return input_ids with shape (1, tokens)")
        if input_ids.shape[1] > MINILM_TOKEN_LIMIT:
            raise ValueError(f"MiniLM input exceeds the 256-token Stage 0 limit: actual {input_ids.shape[1]}")
        features = {key: value.to(self.device) for key, value in features.items()}
        with torch.no_grad():
            output = transformer.auto_model(**features)
        return output.last_hidden_state

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` into `slot_count` vectors of dimension 896.

        Each slot query attends directly to the projected token embeddings
        via scaled dot-product attention. Different queries attend to
        different token positions, so each slot carries a different view
        of the input text.
        """
        token_embeddings = self._token_embeddings(text)  # (1, tokens, 384)
        projected = self.projection(token_embeddings).squeeze(0)  # (tokens, 896)
        scale = self.attn_log_temp.exp()
        attn_logits = self.slot_queries @ projected.T / scale  # (slots, tokens)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (slots, tokens)
        return attn_weights @ projected  # (slots, 896)

    def encode_to_workspace(self, text: str, workspace: Workspace) -> None:
        """Encode `text` and write the result into `workspace`'s slots."""
        slots = self.encode(text)
        workspace.write_slots(slots)
