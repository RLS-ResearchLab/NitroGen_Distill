# Distillation roadmap

The teacher costs `16 DiT evaluations x (8 layers x 1024 width)` per action chunk, plus one
SigLIP2-large pass per frame. Planned strategies, in intended order of implementation:

## 1. Step distillation of the flow head (highest leverage, no architecture change)
- **Consistency distillation** — student maps any point on the teacher's ODE trajectory
  to its endpoint. Data: dataset frames; teacher trajectories generated on the fly with
  `get_action` (or cached). Target: 1–4 step sampling.
- **Progressive distillation** — halve step count repeatedly (16→8→4→2→1); each round the
  student matches two teacher steps with one.
- **ReFlow** — regenerate (noise, action) couplings with the teacher, retrain on the
  straightened pairs so one Euler step suffices.

Implementation hooks: `NitroGenModel.get_action(num_inference_timesteps=N)`,
`NitroGenModel._predict_velocity(...)` for arbitrary-timestep teacher velocity queries.

## 2. Teacher-as-labeler output distillation
Run the teacher over dataset frames; train a smaller student on teacher velocities
(velocity matching at sampled timesteps) and/or sampled action chunks. Doubles as label
denoising, since raw dataset labels are noisy (overlay parsing, controller mapping).

## 3. Vision encoder distillation
SigLIP2-large (24 layers, width 1024, 256 tokens/frame) → smaller ViT. Feature-match the
tokens the action head actually consumes (post-`vl_self_attention_model` embeddings),
not generic image features.

## 4. DiT shrinking with intermediate-feature matching
Fewer blocks / narrower width; match hidden states via
`DiT.forward(..., return_all_hidden_states=True)` alongside the output loss.

## 5. (Aggressive) deterministic regression head
Single forward pass, no flow. Fastest, but loses action multimodality — expect mushy
averaged behavior; use as a baseline, not the main path.

## Known risks
- Offline-only distillation cannot correct compounding rollout drift (no DAgger).
  Mitigation: perturb observations/action histories during training.
- The model's 21-button order is confirmed (official play.py), but the rule that derived
  the four virtual right-stick buttons in the training labels is not published — the
  threshold in `dataset_to_model_actions` is a guess. Validate with open-loop probing
  (`ngd/eval`) on labeled chunks before training on converted labels.
