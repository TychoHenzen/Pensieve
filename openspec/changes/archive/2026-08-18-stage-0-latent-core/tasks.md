## 1. Workspace and concept slots

- [x] 1.1 Implement the workspace data structure: a tensor of shape (N, 768) where N is configurable (default 16), supporting the ablation set {1, 4, 8, 16, 32, 64}
  <!-- covers: workspace/concept-slots -->
  <!-- status: completed -->
- [x] 1.2 Implement snapshot() and restore() for the workspace with bitwise-identical round-trip, and write a test that verifies the round-trip property
  <!-- covers: workspace/concept-slots -->
  <!-- status: completed -->
- [x] 1.3 Implement variance and covariance reporting on the workspace slots for collapse detection, and wire into the instrumentation schema
  <!-- covers: workspace/concept-slots -->
  <!-- status: completed -->

## 2. GSM8K generator

- [x] 2.1 Implement a `gsm8k` generator that wraps the Hugging Face GSM8K dataset as a seeded deterministic stream of Observe (word problem) and Probe (numerical answer) events, conforming to the v1 stream schema
  <!-- covers: eval/generators/gsm8k -->
  <!-- status: completed -->
- [x] 2.2 Verify truth never leaks into the subject view, chance rate reports 0.0, and two runs with the same seed produce the same stream
  <!-- covers: eval/generators/gsm8k -->
  <!-- status: completed -->

## 3. Latent loop

- [x] 3.1 Load Pythia-160M and implement the hidden-state feedback loop: read slot vectors as input embeddings, run a forward pass, project hidden states back to slot space via a learned projection, write them back to slots. One latent step per iteration, configurable step count (default 8)
  <!-- covers: core/latent-loop -->
  <!-- status: completed -->
- [x] 3.2 Implement cost counter mapping: each latent step increments steps by 1 and flops by the forward-pass flop count of Pythia-160M on a sequence of length equal to the slot count
  <!-- covers: core/latent-loop -->
  <!-- status: completed -->
- [x] 3.3 Verify no intermediate token decoding occurs during the latent loop (hidden states feed back directly, not through the vocabulary)
  <!-- covers: core/latent-loop -->
  <!-- status: completed -->

## 4. Codecs

- [x] 4.1 Implement the encoder: frozen all-MiniLM-L6-v2 producing token-level embeddings, a learned linear projection (384 to 768), and a learned cross-attention module with N query vectors (one per slot) that maps projected embeddings to workspace slots
  <!-- covers: codecs/encoder -->
  <!-- status: completed -->
- [x] 4.2 Implement the decoder: feed workspace slot vectors as input embeddings to Pythia-160M's forward pass, decode tokens autoregressively from the LM head logits
  <!-- covers: codecs/decoder -->
  <!-- status: completed -->
- [x] 4.3 Verify encoder weights (all-MiniLM-L6-v2) are frozen and decoder weights (Pythia-160M pretrained) are frozen, while the cross-attention and projection layers are trainable
  <!-- covers: codecs/encoder, codecs/decoder -->
  <!-- status: completed -->

## 5. Subject assembly

- [x] 5.1 Wire the workspace, latent loop, encoder, and decoder into a v1-compliant subject: observe() runs encoder then latent loop, answer() runs latent loop then decoder, idle() is a no-op, snapshot()/restore() delegates to workspace, cost() returns cumulative counters
  <!-- covers: core/latent-loop, workspace/concept-slots, codecs/encoder, codecs/decoder -->
  <!-- status: completed -->
- [x] 5.2 Pass the harness probe isolation test: snapshot, feed noise, restore, verify outputs match a run that never saw the noise
  <!-- covers: core/latent-loop, workspace/concept-slots -->
  <!-- status: completed -->

## 6. Training

- [x] 6.1 Implement the training loop: frozen Pythia-160M backbone, frozen sentence encoder, language modeling loss on GSM8K answer tokens given workspace state after encoding + latent loop. Trainable parameters: cross-attention mapping, hidden-state projection, adaptation layer if added
  <!-- covers: core/latent-loop, codecs/encoder, codecs/decoder -->
  <!-- status: completed -->
- [x] 6.2 Implement VICReg collapse guards: variance term penalizing low per-dimension variance across a batch, covariance term penalizing high inter-dimension covariance, EMA target encoder with configurable decay (default 0.99), stop-gradients on the target side
  <!-- covers: workspace/concept-slots -->
  <!-- status: completed -->
- [x] 6.3 Run training on GSM8K train split (~7.5k problems), verify loss decreases, and check that collapse detection metrics stay healthy
  <!-- covers: core/latent-loop, workspace/concept-slots, eval/generators/gsm8k -->
  <!-- status: completed -->

## 7. Narration decoder

- [x] 7.1 Implement the narration decoder: a linear projection from workspace slot vectors to short token sequences decoded via the Pythia LM head, reading but not modifying the workspace. Off by default, toggled via run config
  <!-- covers: codecs/narration -->
  <!-- status: completed -->
- [x] 7.2 Verify narration produces distinct text for distinct workspace states and writes to the narration field on the run record
  <!-- covers: codecs/narration -->
  <!-- status: completed -->

## 8. Gate evaluation

- [x] 8.1 Run the token chain-of-thought baseline on Pythia-160M over the GSM8K test split. If accuracy is below 5%, substitute a simpler dataset and update the gate definition
  <!-- covers: eval/generators/gsm8k, core/latent-loop -->
  <!-- status: completed -->
- [x] 8.2 Run the latent reasoning subject on the GSM8K test split with at least 5 seeds, compare accuracy against the token-CoT baseline
  <!-- covers: core/latent-loop, eval/generators/gsm8k -->
  <!-- status: completed -->
- [x] 8.3 Run the slot count ablation: {1, 4, 8, 16, 32, 64} slots, compare single-vector (1 slot) against multi-slot versions
  <!-- covers: workspace/concept-slots -->
  <!-- status: completed -->
- [x] 8.4 Run the cycling sweep on FashionMNIST alongside MNIST: measure whether the latent-reasoning architecture handles re-exposure better than replay
  <!-- covers: workspace/concept-slots -->
  <!-- status: completed -->
- [x] 8.5 Write the gate report and pass/fail determination
  <!-- covers: eval/generators/gsm8k, core/latent-loop, workspace/concept-slots -->
  <!-- status: completed -->

## 9. Validation

- [x] 9.1 Run `openspec validate "stage-0-latent-core" --strict --no-interactive` and confirm the change passes
  <!-- covers: core/latent-loop, workspace/concept-slots, codecs/encoder, codecs/decoder, codecs/narration, eval/generators/gsm8k -->
  <!-- status: completed -->
