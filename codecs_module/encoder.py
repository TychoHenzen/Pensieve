from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from workspace.concept_slots import SLOT_DIM, Workspace

MINILM_DIM = 384
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NUM_ATTENTION_HEADS = 8


class SlotEncoder(nn.Module):
    """Encodes text into workspace slot vectors.

    A frozen pretrained sentence encoder (all-MiniLM-L6-v2) produces
    token-level embeddings. A learned linear layer projects those
    embeddings from 384 to 768 dimensions. A learned cross-attention
    module, with one query vector per slot, then reads the projected
    token embeddings into `slot_count` output vectors of dimension 768.
    """

    def __init__(self, slot_count: int = 16, device: str = "cpu") -> None:
        super().__init__()
        self.slot_count = slot_count
        self.device = device

        self._sentence_model = SentenceTransformer(MINILM_MODEL_NAME, device=device)
        for param in self._sentence_model.parameters():
            param.requires_grad = False
        self._sentence_model.eval()

        self.projection = nn.Linear(MINILM_DIM, SLOT_DIM)
        self.slot_queries = nn.Parameter(torch.randn(slot_count, SLOT_DIM))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=SLOT_DIM, num_heads=NUM_ATTENTION_HEADS, batch_first=True
        )

        self.to(device)

    def _token_embeddings(self, text: str) -> torch.Tensor:
        """Frozen token-level embeddings for `text`, shape (1, tokens, 384)."""
        transformer = self._sentence_model[0]
        tokenizer = transformer.tokenizer
        features = tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True
        )
        features = {key: value.to(self.device) for key, value in features.items()}
        with torch.no_grad():
            output = transformer.auto_model(**features)
        return output.last_hidden_state

    def encode(self, text: str) -> torch.Tensor:
        """Encode `text` into `slot_count` vectors of dimension 768."""
        token_embeddings = self._token_embeddings(text)  # (1, tokens, 384)
        projected = self.projection(token_embeddings)  # (1, tokens, 768)

        queries = self.slot_queries.unsqueeze(0)  # (1, slot_count, 768)
        attended, _ = self.cross_attention(
            query=queries, key=projected, value=projected
        )
        return attended.squeeze(0)  # (slot_count, 768)

    def encode_to_workspace(self, text: str, workspace: Workspace) -> None:
        """Encode `text` and write the result into `workspace`'s slots."""
        slots = self.encode(text)
        workspace.write_slots(slots)
