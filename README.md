# llama-rloo-reasoning

From-scratch policy-gradient implementations, building up from classical REINFORCE on CartPole to an RLOO + PPO-clip fine-tune of `Llama-3.2-1B-Instruct` for `<think>…</think>` reasoning on GSM8K, and then sideways to self-play REINFORCE on chess. The same gradient estimator runs through all three notebooks; only the environment, baseline, and policy class change.

The Project 1 notebook is the headline: it's a miniature, didactic version of the recipe behind DeepSeek-R1-style reasoners — rule-based reward, no preference data, no learned reward model.

## Contents

| File | Topic |
|------|-------|
| `RL_cartpole_REINFORCE.ipynb` | REINFORCE with a learned value baseline on CartPole-v1 (PyTorch + Gymnasium). |
| `Project1.ipynb` | RLOO + PPO-clip + KL penalty on `Llama-3.2-1B-Instruct` for GSM8K reasoning. **Main result.** |
| `RL_chess_REINFORCE.ipynb` | Self-play REINFORCE on chess; same estimator as the CartPole notebook applied to a much larger action space. |
| `data/training_report_{1,2,3}.txt` | Raw outputs of the three Project 1 ablation runs summarised below. |
| `cartpole_initial.gif`, `cartpole_final.gif` | Pre/post-training rollouts of the CartPole policy. |

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* — http://incompleteideas.net/book/RLbook2020.pdf (Ch. 3 covers MDPs, Ch. 13 covers REINFORCE).
- HuggingFace TRL — RL algorithms applied to LLMs: https://huggingface.co/docs/trl/index
- DeepSeekMath / GRPO paper: https://arxiv.org/pdf/2402.03300
- REINFORCE Leave-One-Out (RLOO): https://openreview.net/pdf?id=r1lgTGL5DE

The model used throughout Project 1 is `meta-llama/Llama-3.2-1B-Instruct`.

---

## Project 1 — Enforcing Agentic Reasoning (`Project1.ipynb`)

### Objective

Fine-tune `Llama-3.2-1B-Instruct` so that it strictly follows a `<think>…</think>` XML reasoning format on GSM8K math problems, using rule-based RL (no preference data, no learned reward model).

### Algorithm: REINFORCE / RLOO with PPO-style clipping

The notebook implements the training loop from scratch on top of two copies of the same base model:

- **`policy`** — the trainable network whose log-probabilities are differentiated.
- **`ref_policy`** — a frozen copy used only to evaluate `KL(π_θ ‖ π_ref)`. It runs in `eval()` mode with `requires_grad_(False)`, so no optimizer state or backward activations are allocated.

Each training step is split into two phases:

1. **Rollout (no gradients).** For one prompt, `K` completions are sampled in a single batched `generate` call. Each completion is scored by a rule-based reward (well-formed `<think>` block + correct integer answer). Per-token log-probs under the *sampling* policy (`π_old`) and under `π_ref` are cached. The advantage is the **leave-one-out baseline**:
   `A^(i) = r^(i) − (1/(K−1)) · Σ_{j≠i} r^(j)`.
2. **Update.** `PPO_EPOCHS` gradient steps are taken on the cached rollout using a clipped importance-sampling surrogate (`clip(π_θ/π_old, 1−ε, 1+ε)`) and a KL penalty `β · KL(π_θ ‖ π_ref)` estimated with the K3 estimator. The clipping is what makes it safe to reuse the same rollout across multiple epochs — it's the same trick GRPO and most modern LLM-RL stacks rely on.

Reward signal is purely rule-based: a regex check for exactly one `<think>…</think>` block followed by a final answer token, plus an exact-match check on the extracted integer answer against the GSM8K ground truth.

### Evaluation

`check_accuracy` reports three numbers on a held-out slice:

- **Format accuracy** — fraction of completions with a valid `<think>…</think>` structure.
- **Answer accuracy** — fraction whose extracted answer matches the ground truth (regardless of format).
- **Combined accuracy** — fraction satisfying both.

Baseline is measured before any updates; the same prompts are re-run through the trained policy at the end for a paired comparison.

### Empirical results

The `data/` folder collects three training reports produced by `Project1.ipynb`. All three runs share the same model, dataset, optimizer (Adam, `lr = 1e-5`), and budget (100 prompts × 1 RLOO step each). They differ only in the four knobs below, which lets us read the deltas as a small ablation on KL strength, group size, and update aggressiveness.

| Run | K  | KL coef | Clip ε | PPO epochs | Format Δ | Answer Δ | Combined Δ |
|-----|----|---------|--------|------------|----------|----------|------------|
| 1 (`training_report_1.txt`) | 15 | 0.50  | 0.2 | 5 | 46 → 72 (**+26**) | 30 → 32 (**+2**)  | 18 → 28 (**+10**) |
| 2 (`training_report_2.txt`) | 10 | 0.05  | 0.2 | 2 | 51 → 70 (**+19**) | 35 → 46 (**+11**) | 26 → 34 (**+8**)  |
| 3 (`training_report_3.txt`) | 16 | 0.01  | 0.3 | 2 | 51 → **100** (**+49**) | 35 → 39 (**+4**)  | 26 → 39 (**+13**) |

### Discussion

- **KL coefficient is the dominant lever.** Format gains scale almost monotonically as the KL penalty is relaxed (0.50 → 0.05 → 0.01). Run 3, with `β = 0.01`, drives format accuracy to 100% — the policy is essentially free to specialize on the rule-based reward because it is barely anchored to the reference.
- **Format ≠ reasoning.** Run 3 achieves perfect format compliance but only +4 pp on raw answer accuracy. This is the classic *reward-hacking* signature of rule-based RL: the policy learns the cheapest path to the reward (matching the regex) faster than it learns the harder, sparser signal (solving the math). Run 1, with the strongest KL anchor (`β = 0.5`), is the mirror image — it stays close to the reference but learns very little about either objective (+2 pp on answers).
- **Run 2 is the most balanced.** Moderate KL (`β = 0.05`), smaller `K = 10`, and only 2 PPO epochs yield the best **answer-accuracy lift (+11 pp)** of the three configurations and a healthy combined-accuracy gain (+8 pp). The interpretation is that just-enough regularization preserves the model's general reasoning capacity while still letting the format reward propagate.
- **Combined accuracy is the metric to trust.** It penalizes both failure modes (well-formatted nonsense, correct-but-malformed answers). Ranked by combined Δ: Run 3 (+13) > Run 1 (+10) > Run 2 (+8). Ranked by combined *post-training level*: Run 3 (39%) > Run 2 (34%) > Run 1 (28%). Run 3 wins on the headline number, but most of that win comes from format compliance rather than improved reasoning.
- **Caveats.** Each run is a single seed over only 100 prompts; the differences above are suggestive rather than statistically conclusive, and the held-out evaluation slice is small (≤100 questions, integer-valued answer match). A more rigorous study would repeat each configuration over multiple seeds and report confidence intervals on Δ.

### Practical takeaway

If the goal is *format adherence* (e.g. preparing a model to be parsed by a downstream tool), Run 3's recipe — low KL, larger group, moderate clip — is the right shape. If the goal is *reasoning quality* on top of a format constraint, Run 2's milder KL with fewer PPO epochs is the safer starting point, and the next experiments worth running are: (i) a longer schedule at Run 2's hyperparameters, and (ii) Run 3's KL with a curriculum that gates the format reward once compliance saturates, so the optimizer can spend its remaining budget on answer correctness.

---

## Companion notebooks

### `RL_cartpole_REINFORCE.ipynb`

Classic REINFORCE with a learned value baseline on `CartPole-v1`. Two small MLPs (a `PolicyNetwork` outputting a `Categorical` action distribution and a `ValueNetwork` estimating `V_w(s)`) are trained jointly. The `cartpole_initial.gif` / `cartpole_final.gif` pair shows the policy before any training and after convergence.

This notebook exists to make the gradient estimator explicit before scaling up: the same `(G_t − V_w(s_t)) · ∇_θ log π_θ(a_t | s_t)` shape reappears in `Project1.ipynb`, with the value critic replaced by an RLOO baseline (no value head needed for a 1B-parameter LM).

### `RL_chess_REINFORCE.ipynb`

Self-play REINFORCE on chess. The state is a board encoding, the action space is the legal-move set at each position, and the reward signal is the game outcome. This is the same estimator as the CartPole notebook — it just lives in a much larger and harder environment, and it makes concrete why moving to leave-one-out / clipped variants matters once the variance starts dominating.

---

## Reproduction notes

- `Project1.ipynb` was developed on a single 24 GB GPU. With `Llama-3.2-1B-Instruct` in `bfloat16` and `K = 10–16` completions per prompt, peak memory sits around 12–16 GB; smaller `K` lowers the footprint.
- The reference policy is a separate frozen copy of the base model (`requires_grad_(False)`, `eval()` mode). It contributes no optimizer state and no backward activations.
- 100-step training runs take roughly 1–2 hours on the above hardware, depending on `K` and `PPO_EPOCHS`.
- All three Project 1 runs use the same seed-free configuration; the reports in `data/` are single-seed, so treat the deltas as suggestive rather than statistically conclusive.

## License

MIT — see `LICENSE`.
