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
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device

    def decode(self, workspace: Workspace) -> str:
        """Decode `workspace`'s slots into text, without mutating the workspace."""
        slots = workspace.read_slots().to(self.device)
        input_embeds = slots.unsqueeze(0)  # (1, N, hidden_dim)

        embedding_layer = self.model.get_input_embeddings()
        eos_token_id = self.tokenizer.eos_token_id

        generated_ids: list[int] = []

        with torch.no_grad():
            for _ in range(self.max_tokens):
                outputs = self.model(inputs_embeds=input_embeds)
                next_token_logits = outputs.logits[0, -1, :]
                next_token_id = int(torch.argmax(next_token_logits).item())

                if eos_token_id is not None and next_token_id == eos_token_id:
                    break

                generated_ids.append(next_token_id)

                next_token_tensor = torch.tensor(
                    [[next_token_id]], device=self.device
                )
                next_token_embed = embedding_layer(next_token_tensor)  # (1, 1, hidden_dim)
                input_embeds = torch.cat([input_embeds, next_token_embed], dim=1)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
