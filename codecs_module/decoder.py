from __future__ import annotations

import torch

from workspace.concept_slots import Workspace

DEFAULT_MAX_TOKENS = 64


class SlotDecoder:
    """Decodes workspace slot vectors into text via a shared, frozen LM.

    The decoder does not own the language model - it receives it (and its
    tokenizer) as dependencies, shared with `LatentLoop`. Workspace slot
    vectors are fed as input embeddings occupying the first N positions of
    the sequence; the model's existing LM head then produces logits, and
    tokens are decoded autoregressively via greedy (argmax) selection until
    EOS or `max_tokens` is reached. The workspace itself is never modified.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device or next(model.parameters()).device

    def decode(self, workspace: Workspace) -> str:
        """Decode `workspace`'s slots into text, without mutating the workspace."""
        slots = workspace.read_slots().to(self.device)
        embedding_layer = self.model.get_input_embeddings()
        actual_width = _embedding_width(embedding_layer)
        if slots.shape[-1] != actual_width:
            raise ValueError(
                f"expected slot width {actual_width}, actual slot width {slots.shape[-1]}"
            )
        input_embeds = slots.unsqueeze(0)  # (1, N, hidden_dim)
        eos_token_id = self.tokenizer.eos_token_id

        generated_ids: list[int] = []
        past_key_values = None

        with torch.no_grad():
            for _ in range(self.max_tokens):
                outputs = self.model(
                    inputs_embeds=input_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = int(torch.argmax(next_token_logits).item())
                past_key_values = getattr(outputs, "past_key_values", None)

                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                generated_ids.append(next_token_id)

                next_token_tensor = torch.tensor(
                    [[next_token_id]], device=self.device
                )
                input_embeds = embedding_layer(next_token_tensor)  # (1, 1, hidden_dim)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def _embedding_width(embedding_layer: torch.nn.Module) -> int:
    """Read the width exposed by a model's public input embedding module."""
    embedding_dim = getattr(embedding_layer, "embedding_dim", None)
    if isinstance(embedding_dim, int):
        return embedding_dim
    for parameter in embedding_layer.parameters():
        if parameter.ndim >= 2:
            return parameter.shape[-1]
    raise ValueError("could not determine model input embedding width")
