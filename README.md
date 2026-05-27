# cpg-rl-cbo

Modular IsaacLab-oriented skeleton for **RL + CPG + CBO** gait adaptation across multiple quadruped morphologies and terrains.

## What is implemented

- Two RL→CPG action interfaces:
  - `per_leg_mu_omega_phi`: RL outputs `mu, omega, phi` for each leg.
  - `shared_omega_phase_offsets`: RL outputs per-leg `mu`, one shared `omega`, per-leg phase offsets, and shared `phi`.
- Observation profiles:
  - `full`: command velocity, body state, joint state, foot contacts, last action, CPG state.
  - `medium`: full observation without joint state and last action.
- RL→CPG→IK pipeline (`GaitAdaptationPipeline`) with swappable modules.
- Generic IK baseline compatible with Go1, Go2, A1, and Anymal-B morphology specs.
- Terrain and morphology configuration for multi-morphology / multi-terrain training setup.
- CBO parameter schema for optimizing: `g_c_hind`, `g_c_front`, `g_p_hind`, `g_p_front`, `x_offset_front`, `x_offset_hind`, `d_set_front`, `d_step_hind`, `h`.
- CBO conditioning vector from morphology properties: mass, scale, link lengths, COM.
- IsaacLab-ready training/testing CLI scaffolding (see below).

## Package layout

- `src/cpg_rl_cbo/action_modes.py`: RL action decoding modes.
- `src/cpg_rl_cbo/observations.py`: full/medium observation builders.
- `src/cpg_rl_cbo/cpg.py`: oscillator state and foot trajectory generation.
- `src/cpg_rl_cbo/ik.py`: IK abstraction and quadruped implementation.
- `src/cpg_rl_cbo/morphology.py`: morphology and terrain definitions.
- `src/cpg_rl_cbo/cbo.py`: CBO parameter and conditioning schemas.
- `src/cpg_rl_cbo/pipeline.py`: end-to-end modular gait adaptation pipeline.
- `src/cpg_rl_cbo/isaaclab/`: IsaacLab task registry, CLI parsing, and controller bridge.
- `scripts/train.py`: IsaacLab training entrypoint (RSL-RL).
- `scripts/play.py`: IsaacLab evaluation entrypoint.

## IsaacLab training/testing (similar to cpg-rl)

These scripts mirror the structure of the `cpg-rl` repository but are adapted to the multi-morphology gait adaptation setup.

### Dry-run config resolution

```bash
PYTHONPATH=src python scripts/train.py --dry_run --task=multi_morph_train
PYTHONPATH=src python scripts/play.py --dry_run --task=multi_morph_eval
```

### Training

```bash
PYTHONPATH=src python scripts/train.py \
  --task=multi_morph_train \
  --enable_cpg \
  --run_name=cpg_cbo_multi \
  --headless
```

### Evaluation / playback

```bash
PYTHONPATH=src python scripts/play.py \
  --task=multi_morph_eval \
  --enable_cpg \
  --load_run=cpg_cbo_multi \
  --headless
```

### Task registry

`src/cpg_rl_cbo/isaaclab/task_registry.py` defines two default task specs and their IsaacLab task names. Update
`isaaclab_task` to match the task registered in your IsaacLab environment.

## DeepTransition and IsaacLab integration note

This repository now contains the core modular interfaces needed to integrate:

1. A DeepTransition policy head that emits either supported action mode.
2. IsaacLab task wrappers that provide observation tensors matching `ObservationProfile`.
3. Terrain/morphology curriculum sampling around `Terrain` and `MorphologySpec`.
4. CBO outer-loop updates over `CBOParameters` conditioned on `MorphologyCondition`.

## Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
