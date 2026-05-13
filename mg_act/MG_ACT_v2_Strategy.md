# MG-ACT v2 — Strategy & Build Plan for AIC Qualification

A 7-day plan to get from your current state (≈500 episodes collected, MG-ACT proposal drafted, L4 Colab access) to a submitted, scoring-competitive policy. Read top to bottom once, then use as a working doc.

---

## 1. What changes after looking at your actual data

You shared one episode (`episode_1778199680953_ff0e6d8c.h5`, 530 steps, 31.7 s, success=True, SFP→sfp_port_0 on nic_card_mount_0). Three things in that file alter the architecture doc materially.

**1.1. Logged rate is 16.7 Hz, not 100 Hz.** Your file attribute is `control_hz = 16.725574…`. Force/torque, joint state, and images are all logged at the same rate, perfectly synchronous. The architecture doc reasons about a 100 Hz haptic stream cross-attending into a 15 Hz visual stream — that misalignment problem does not exist in your dataset as collected. Two implications:

- The "Cross-Modal Attention with temporal upsampling via latent frame injection" section becomes overengineering for what you have. Vision tokens and haptic tokens align step-for-step. Use a much simpler fusion (bidirectional cross-attention over time, no rate-bridging).
- If you truly want high-rate haptic context, log a haptic ring buffer at 100 Hz on the side and slice a recent window per timestep. But this is optional, and on L4 with 7 days I would not chase it for v1.

**1.2. Your action layout is already exactly the variable-impedance vector the architecture doc proposes.** From the file's `@action_layout` attribute: `translation[3] | quaternion_wxyz[4] | stiffness_diag[6] | damping_diag[6]` = 19 dims. That's the action space — keep it. You are predicting compliance directly, which is the whole point of the design.

**1.3. The control loop you actually need is ≤ 20 Hz, not 100 Hz.** The AIC docs are explicit: `aic_adapter` delivers the synchronized `Observation` at up to 20 Hz, and the policy issues targets to `aic_controller` (which itself runs the 500 Hz inner loop). So when the architecture doc talks about "100 Hz control loop demanded by the UR5e", that's not the policy's loop — that's the controller's. Your policy budget is one inference per ~50 ms, which is generous and lets you use temporal ensembling normally.

**1.4. Cosmos isn't in your data and probably isn't worth the engineering cost in 7 days.** Your H5 stores raw `uint8` RGB at 256×288 (already image_scale 0.25 from 1024×1152). Wiring up Cosmos-Predict embeddings means another inference dependency at runtime, more latency, and a much larger memory footprint on L4 (24 GB). For Phase 1 qualification, a per-camera ResNet-18 trained from ImageNet, or a frozen DINOv2-small, gets you 90% of the visual-feature benefit at 5% of the complexity. Save Cosmos for Phase 2 if you advance.

**1.5. References worth re-checking.** Two cited papers in your doc are real and load-bearing — keep them: ACT (Zhao et al., 2023), ReTac-ACT (Ruan et al., 2026, arXiv 2603.09565). The "Cosmos Policy" 2026 reference exists as an NVIDIA cookbook recipe but isn't a peer-reviewed methodology paper, so weight it accordingly. The "Causal Confusion in Imitation Learning" paper (de Haan et al.) is real and the modality-dropout justification still stands.

A new one worth knowing about: **PhaForce** (Wang et al., March 2026, arXiv 2603.08342) — phase-scheduled visual-force policy with an explicit slow-fast architecture. It hits 86% on contact-rich USB-style insertion with a +40 pp lead over RDP / Diffusion Policy / Force-Concat. We're going to borrow the *contact-aware phase* idea from it as an auxiliary head, but not the full slow-fast decomposition (too much engineering for 7 days).

---

## 2. Recommended architecture — call it MG-ACT v2

The MG-ACT skeleton you proposed is sound. Below is the version I'd actually build, with the changes from §1 applied. Everything here is L4-trainable in bf16 with batch size 32 and 256×288 images.

### 2.1. Inputs (per timestep t, plus a short window)

| Stream | Shape | Source in your H5 | Notes |
|---|---|---|---|
| Images | 3 × (256, 288, 3) uint8 | `observations/images/{left,center,right}` | Resize→224×224 for backbone; standard ImageNet normalization. |
| Wrench window | (T_h=8, 6) float32 | `observations/wrench` last 8 steps | ~0.5 s of haptic context at 16.7 Hz. |
| Joint state | (7,) + (7,) + (7,) | `observations/{joint_position, joint_velocity, joint_effort}` | 21-D proprio. |
| (Optional) TCP pose | (7,) | not in this episode but available via `controller_state.tcp_pose` at runtime | Helps a lot — log it next time you collect. |

### 2.2. Targets (per timestep, predicted as a chunk of length k)

| Sub-action | Source | Representation in network | Notes |
|---|---|---|---|
| translation | `actions/translation` | regress directly (3-D) | Frame is `base_link`. |
| rotation | `actions/quaternion_wxyz` | **predict as 6-D continuous** (Zhou et al., 2019) | Quaternion regression has a discontinuity (q ≡ −q) that hurts learning; convert quat → 6D for training, regress 6D, then map back to quat for the ROS message. PyTorch3D has the helpers. |
| stiffness | `actions/stiffness_diag` | regress in **log-space**, clip to a safe range | log10(K) ∈ [1, 3] roughly; reduces dynamic range. |
| damping | `actions/damping_diag` | regress in **log-space** | Same reason. |

So the network output per step is 21-D (3 + 6 + 6 + 6); the lifecycle wrapper converts back to the 19-D ROS payload before publishing.

### 2.3. Network — modules

```
                         ┌──────────────┐
   left  ──ResNet18 ────▶│              │
   ctr   ──ResNet18 ────▶│  Visual      │── Vis tokens (T_v, D)
   right ──ResNet18 ────▶│  Tokenizer   │
                         └──────────────┘
                         ┌──────────────┐
   wrench[T_h,6] ───────▶│ 1D-Conv +    │── Hap tokens (T_h, D)
                         │ MLP          │
                         └──────────────┘
                         ┌──────────────┐
   joint_pos             │              │
   joint_vel    ────────▶│ Proprio MLP  │── proprio embed s_t (D,)
   joint_effort          │              │
                         └──────────────┘

              ┌────────────────────────────────────┐
              │  Bidirectional Cross-Attention     │
   Vis ──────▶│   Vis ↔ Hap                        │── Fused tokens
   Hap ──────▶│                                    │
              └────────────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────────────┐
              │  Proprio-conditioned gate          │
              │  g = σ(MLP(s_t)) ∈ [0, 1]^D        │
              │  fused = g·hap_ctx + (1-g)·vis_ctx │
              └────────────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────────────┐
              │  ACT Encoder (Transformer)         │
              │  + CVAE latent z (training only)   │
              └────────────────────────────────────┘
                            │
                            ▼
              ┌────────────────────────────────────┐
              │  ACT Decoder (Transformer)         │
              │  Cross-attends to fused tokens     │
              │  Outputs chunk of k actions (21-D) │
              └────────────────────────────────────┘
```

A few important details:

- **Vision encoder.** Start with a single shared ResNet-18 (not per-camera) initialized from ImageNet, fine-tuned. Concatenate the three camera tokens with a learned camera-id embedding. This is the LeRobot-ACT default and the cheapest thing that works. If overfitting, try freezing early blocks. If underfitting, switch to DINOv2-small (frozen, ViT-S/14 outputs are 384-D and well-regularized).
- **Haptic encoder.** Three Conv1D layers (kernel 3, stride 1, channels 32→64→D) over the (T_h, 6) wrench window, then a small MLP per timestep. Don't skip this — even the synchronous-rate dataset benefits because Conv1D extracts local force *transients* that a single-step MLP can't see.
- **Bidirectional cross-attention.** ReTac-ACT's mechanism. One block for Vis attends to Hap, one block for Hap attends to Vis, both produce updated tokens. ~2 layers each is plenty.
- **Proprio-conditioned gate.** Single MLP `s_t → g`. The training signal that makes `g` meaningful comes from the modality-dropout regularizer below. This is the "MG" in "MG-ACT" and it's the right primitive.

### 2.4. Modality dropout — improve on the architecture doc

The doc proposes "fixed dropout in the final 10% of trajectory". Two upgrades I'd make:

1. **Contact-conditioned dropout (primary).** During training, when `‖wrench_xyz‖_2 > τ` (start with τ=5 N), mask out vision with probability 0.5. This directly targets the failure mode (occlusion at insertion) without assuming "10%" is the right fraction. You have wrench per timestep, so the mask is computable at dataset-load time.
2. **Random vision dropout (secondary).** Independent Bernoulli p=0.1 every step regardless of contact, just to prevent the network from co-adapting to always-on vision. Cheap and standard.

Don't drop *both* modalities. Don't drop haptic (you want the gradient through the haptic path always available).

### 2.5. Auxiliary losses — these are where you buy SOTA

The action L1 loss + CVAE KL is the ACT default. Add:

- **F/T reconstruction loss (`λ_recon=0.1`).** A small head off the fused tokens reconstructs the wrench window. ReTac-ACT showed this forces the haptic pathway to actually encode contact information rather than degenerate to a passthrough. Cheap.
- **Contact-phase classification (`λ_phase=0.05`).** Free-space / approach / contact / inserted, derived weakly from `‖wrench‖` and trajectory progress. PhaForce showed this helps the policy schedule force usage. Even without their full slow-fast architecture, the phase head alone is a useful regularizer.
- **Smoothness penalty on stiffness/damping chunks (`λ_smooth=0.01`).** L2 on the 1st difference along the chunk. Discourages compliance toggling, which the AIC scoring's jerk metric rewards.

So total loss is roughly: `L = L1(action) + β·KL(z) + 0.1·MSE(wrench_recon) + 0.05·CE(phase) + 0.01·smooth(K, D)`.

### 2.6. Inference behavior

- Chunk length **k=32** (~1.9 s at 16.7 Hz). The original ACT paper uses 50 at 50 Hz; you're at a third the rate so a smaller chunk is appropriate.
- **Temporal ensembling** (TE): exponential weighting m=0.01 per ACT default. Crucial for jerk score.
- Run inference at ~10 Hz (every 3rd observation), publish with TE-smoothed action. This buys you headroom on Colab→cloud-eval latency drift and the controller does the high-rate work.

---

## 3. Mapping AIC scoring to architectural choices

A quick crosswalk so you can sanity-check the design against what's actually being measured. The full scoring is in `docs/scoring.md`; the highlights:

- **Tier 1 (validity, 1 pt):** Lifecycle node behaves correctly. This is plumbing, not modeling. See §5.
- **Tier 2 trajectory smoothness (0–6 pts):** Average linear jerk. → Action chunking + temporal ensembling + the smoothness penalty in §2.5.
- **Tier 2 task duration (0–12 pts):** Faster = better, capped at 5 s. → Don't add unnecessary settle/wait phases. Penalize dwell in your demonstrations.
- **Tier 2 trajectory efficiency (0–6 pts):** Cumulative path length. → The smoothness + L1 loss already encourage short paths; nothing else needed.
- **Tier 2 insertion force penalty (0 to −12 pts):** > 20 N for > 1 s. → Variable-impedance prediction (you already do this) + contact-conditioned dropout + the phase head all push toward gentle insertion.
- **Tier 2 off-limit contact (0 to −24 pts):** Forearm hits the enclosure walls. → Critical: add a workspace-clamping wrapper *outside* the network, even if the network has implicitly learned the workspace. See §5.
- **Tier 3 success (−12 to 75 pts):** Correct-port insertion. → This is the dominant signal. Train to maximize success first, then optimize Tier 2.

Note the asymmetry: a wrong-port insertion is **−12** while no insertion is 0–25 (proximity score). So a confident wrong policy is *worse* than a hesitant right-direction policy. The CVAE latent + temporal ensembling work in your favor here — they avoid mode-collapse onto one wrong port.

---

## 4. The 7-day execution plan

Today is Day 0. Submission deadline is ~Day 7. I've assumed you have the eval container working and one teammate continuing data collection.

### Day 0 (today)

- [ ] Drop `mg_act_v2_skeleton.py` into a Colab notebook, point it at the one H5 file, run a single forward+backward pass to verify shapes and gradient flow. End-to-end smoke test before anything else.
- [ ] Write a streaming HDF5 dataset class that reads from a list of episode files. Validate by training for ~50 steps on the one episode (overfit-on-one) — loss should drop fast. If it doesn't, the model is broken.
- [ ] Confirm 19-D action layout matches across all collected episodes (check `@action_layout` attr on every file). If anyone changed it during collection, you have a normalization headache.
- [ ] Write the rotation conversion helpers (quat → 6D for training, 6D → quat for ROS publish) and unit-test them.

### Day 1

- [ ] Train a **vision-only ACT baseline** on the full collected dataset. No haptic, no fancy fusion, no aux losses. This is your floor — if MG-ACT v2 doesn't beat this on validation success rate, something is wrong. Should train in ~3-4 hours on L4.
- [ ] Set up evaluation harness: pull one held-out episode, replay observations through the policy, log predicted vs. expert actions, compute open-loop MSE. (Closed-loop sim eval comes later via the docker compose route.)
- [ ] Continue data collection in parallel — every additional episode helps.

### Day 2

- [ ] Train **MG-ACT v2 (no aux losses)** — full architecture, but only the action L1 + KL loss. Compare val loss and open-loop MSE to vision-only baseline. The cross-modal architecture should beat vision-only by 10–20% on action MSE; if it doesn't, debug.
- [ ] Set up the AIC submission docker locally per `docs/submission.md`. Verify with `docker compose up`. Don't submit yet; you only get 1 submission/day.

### Day 3

- [ ] Train **MG-ACT v2 full** — all aux losses on, contact-conditioned dropout on. Save best-by-val-loss checkpoint.
- [ ] Closed-loop sim eval against the eval container locally. Aim for >50% success on the 3 trial types. Iterate on hyperparameters: chunk size (try 24, 32, 48), `β` (try 5, 10, 20), `τ` for contact dropout (try 3, 5, 8 N).

### Day 4

- [ ] Make the **first submission** — even a mediocre score gets you on the leaderboard and validates the whole submission pipeline. The risk of leaving submission until Day 7 is that ECR auth, container boot timeouts, and ROS lifecycle bugs eat hours.
- [ ] Train a **second variant** in parallel: stronger augmentations (color jitter ±0.2, random crop 224 from 240, gaussian noise on wrench σ=0.5 N), longer training (200K steps).

### Day 5

- [ ] Ablate: which aux loss is actually helping? Drop each in turn, retrain for 50K steps, measure. You want to know what's load-bearing in case you have to deploy a smaller variant.
- [ ] Investigate failures from Day 4 submission. Common ones: wrong port (mode collapse), insertion force penalty (compliance not adapting), off-limit contacts (workspace clamp missing or buggy).

### Day 6

- [ ] **Final training run** with everything that worked. Train to convergence (lowest val loss) — don't use the final checkpoint, use the best.
- [ ] Re-test docker container locally. Run all 3 trials and inspect `scoring.yaml`.
- [ ] Add the safety wrappers explicitly (workspace clamp, max-force watchdog, time budget) — see §5.

### Day 7

- [ ] **Final submission.** Tag a fresh image (you can't overwrite tags on ECR per `docs/submission.md`). Watch the dashboard.
- [ ] If anything fails at Tier 1 (lifecycle), you have 1 retry. Don't burn it on a code bug — diff carefully against the previous successful submission.

---

## 5. AIC compliance gotchas

These are the things that will surprise you at submission time. From `docs/challenge_rules.md` and `docs/troubleshooting.md`:

**Lifecycle behavior is enforced.** Your `aic_model` node must:
- Start in `unconfigured`, publishing nothing.
- Transition to `configured` within 60 s. Model loading (PyTorch checkpoint download, JIT compile, etc.) happens here.
- Transition to `active` within 60 s. Now the `/insert_cable` action server starts accepting goals.
- Goals must complete within `task.time_limit` (sim time, not wall-clock).
- Cleanup/shutdown must work without leaving publishers active.

If you use the provided `aic_model` framework (recommended), most of this is plumbing that you inherit. The thing to watch is **heavy top-level imports**:

> *"When the policy module is loaded, all top-level code, including imports, runs within a 30-second model discovery budget. Importing large libraries such as `torch` at the top of the module can exceed this budget and cause the policy to be killed before it reports an error."*

So your `MGActV2Policy.__init__` is where you import torch/transformers and load the checkpoint. The class body must stay light.

**Use ROS sim-time, not `time.time()`.** From the troubleshooting doc:
> *"The task time limit is measured against simulation time (the ROS clock), not wall-clock time. A policy that relies on `time.time()` may work locally but misbehave on the portal."*

`self.time_now()` and `Duration(seconds=...)` from rclpy are the right APIs.

**Don't subscribe to ground-truth topics.** During eval, `ground_truth:=false`, and the access control list (Zenoh ACL) blocks `/gz_server/*`, `/scoring/*`, etc. anyway. The only correct sources are the topics in `docs/aic_interfaces.md`.

**Workspace clamp.** The off-limit contact penalty is −24 per trial. A network that's *almost* right can still wander into walls. Add a hard clamp on translation targets *after* the network output, before publishing the `MotionUpdate`. Approximate the safe box from `docs/task_board_description.md` and the URDF; conservatively e.g. `x ∈ [-0.7, 0.0]`, `y ∈ [-0.5, 0.5]`, `z ∈ [0.05, 0.5]` in `base_link`. Tune against your dataset's empirical bounds.

**Stiffness/damping safety clamp.** Cap predicted stiffness at e.g. 500 N/m and damping at e.g. 80 Ns/m. The network can briefly diverge during distribution shift; capping prevents an aggressive command from destabilizing the controller.

**One submission per day.** Don't waste them. Use docker compose locally for verification first.

---

## 6. Things explicitly *not* in v1

So you can defer them with a clear conscience:

- **Cosmos-Predict embeddings.** Phase 2 if you advance.
- **Multi-rate fusion (100 Hz haptic).** The data isn't logged that way. Add only if v1 underperforms specifically on contact-phase failures.
- **Diffusion Policy variant.** With 500 demos, ACT is competitive with DP. With 1000+, DP starts to win on precision insertion. If you have time on Day 6 you could try a Diffusion Policy v2 as a Plan B, but don't make it the primary.
- **Fast residual corrector (PhaForce-style).** The ROS lifecycle and rate constraints make a true two-rate architecture nontrivial. The phase-prediction *signal* is enough for v1.
- **VLA / language conditioning.** No language input in the AIC task spec.
- **End-to-end RL fine-tuning.** No simulator-in-the-loop training infrastructure for 7 days; you're an imitation learning team this week.

---

## 7. Quick training-recipe reference

| Knob | Starting value | Notes |
|---|---|---|
| Optimizer | AdamW | lr 1e-4 (transformer), 1e-5 (ResNet), weight_decay 1e-4 |
| Schedule | cosine, 5% warmup | 100K–200K steps |
| Batch size | 32 | image-heavy; if OOM, drop to 16 + grad-accum 2 |
| Mixed precision | bf16 | L4 supports bf16 natively |
| Chunk size k | 32 | ~1.9 s at 16.7 Hz |
| TE weight m | 0.01 | ACT default |
| KL weight β | 10 | sweep [5, 10, 20] |
| Recon weight | 0.1 | wrench reconstruction |
| Phase weight | 0.05 | weak labels from wrench thresholds |
| Smooth weight | 0.01 | on K, D chunks |
| Contact threshold τ | 5 N | for vision-dropout trigger |
| Vision dropout p | 0.1 | random, in addition to contact-cond |
| Image augs | color jitter ±0.2, RandomCrop 224 from 240, mild blur | helps generalization to randomized board pose |
| Wrench augs | gaussian noise σ=0.3 N | accounts for sim-eval F/T noise floor |

Save the **best-by-eval-loss** checkpoint. The final-epoch checkpoint is almost always overfit by a noticeable margin.

---

## 8. What I'd worry about most

1. **Wrong-port insertions.** With multiple ports visible in trial 1/2 NIC scenarios and only `task.target_module_name` distinguishing them, the policy can mode-collapse onto an off-target port. Mitigation: include `target_module_name` and `port_name` as a one-hot or learned embedding in the encoder input. Look at `aic_task_interfaces/msg/Task.msg` for the exact field names — feed them into the network.
2. **Sim-to-eval distribution shift.** Your collection is presumably with `ground_truth:=true`. The eval is `ground_truth:=false`. The vision distributions are identical, but the *behavior* of your CheatCode-style demos may not be. Audit a few demos manually and make sure the trajectories don't rely on any quirk of the ground-truth path.
3. **Cable tracking.** The task is *cable* insertion. The cable is deformable — it can swing, twist, jam. Your cameras don't see the plug-port interface clearly during the last few millimeters. This is the regime where the haptic path has to take over, and where the design earns its keep. If v1 fails specifically here, the fix is more contact-conditioned dropout aggressiveness (drop vision earlier and more often), not more vision data.
4. **Lifecycle timing.** As above: a top-level torch import will silently kill your container at 30 s with no logs. Test the lifecycle path locally before the leaderboard does it for you.
