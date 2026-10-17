# Compose Your Oracles: Off Policy Improvement with Aggregated Guidance

Code for the NeurIPS 2026 paper *Compose Your Oracles: Off Policy Improvement with Aggregated Guidance* (OPI) by Jingtian Ji, Xuefeng Liu, and Matthew R. Walter.

## Installation

Run these steps from the repository root.

```bash
conda create -n opi python=3.10 -y && conda activate opi
pip install torch==2.9.1      # pick the build that matches your CUDA version
pip install swig
pip install -r requirements.txt

git clone https://github.com/Farama-Foundation/Metaworld.git
cd Metaworld && git checkout 066c391a
git apply ../patches/metaworld_pick_place.patch
pip install -e . && cd ..
```

The patch changes the Pick-Place success threshold and reward used in the paper.
Rendering uses `MUJOCO_GL=egl` by default; set `MUJOCO_GL` to another backend if EGL is unavailable.

## Code layout

| Path | Contents |
|---|---|
| `algo/oracles/` | OPI and oracle loading |
| `algo/baselines/` | Max-Q / Max-V (`maxVQ.py`), LOKI, MAPS, CUP, QMP (`qmix.py`) |
| `model/` | Simba-v2 networks and oracle wrappers |
| `env/` | MetaWorld, DMC, and customized Box2D environments |
| `data_buffer/` | Replay buffer |
| `scripts/` | Training entry points |
| `config/` | Hydra configs for the main results and ablations |
| `third_party/` | Customized Box2D environments and scripted MetaWorld oracles |

## Oracles

MetaWorld oracles are scripted and need no checkpoints.
DMC and Box2D oracle checkpoints will be released separately; their expected paths are listed in `config/oracles/` relative to the project root, which defaults to the current directory and can be set with `export ORACLES_PROJECT_ROOT=/path/to/OPI`.

## Running

Run all commands from the repository root.
Logging uses Weights & Biases (optionally set `WANDB_ENTITY`); add `training.use_wandb=false` to disable it.
Training runs on a GPU when one is visible; CPU-only runs work but are much slower.
The paper uses seeds 42 to 46.

**OPI**

```bash
# MetaWorld; TASK in assembly, drawer-open, hammer, pick-place, push-wall (sparse reward)
# or basketball, lever-pull, peg-insert-side, peg-unplug-side, stick-pull (drop env.sparse_reward=true)
python -m scripts.train_OPI --config-name metaworld/OPI_${TASK} \
    env.sparse_reward=true training.total_timesteps=500000 training.capacity=500000 seed=${SEED}

# DMC; TASK in hopper, quad, walker, cheetah, cartpole
python -m scripts.train_OPI --config-name dmc/OPI_${TASK} \
    training.total_timesteps=500000 seed=${SEED}

# Box2D; TASK in bipedal, lander, racingcar, weather
python -m scripts.train_OPI_simba --config-name box2d/OPI_${TASK}_learner seed=${SEED}
```

**Baselines.** `ALGO` is one of `cup`, `loki`, `maps`, `qmix` (QMP), or `maxVQ` (Max-V; add `training.selection_mode=max_q` for Max-Q).

```bash
python -m scripts.baselines.train_${ALGO} --config-name baselines_configs/${ALGO}/metaworld/${TASK} \
    env.sparse_reward=true training.total_timesteps=500000 training.capacity=500000 seed=${SEED}
python -m scripts.baselines.train_${ALGO} --config-name baselines_configs/${ALGO}/${TASK}/${ALGO}_${TASK} \
    training.total_timesteps=500000 seed=${SEED}                                  # DMC
python -m scripts.baselines.train_${ALGO}_simba --config-name baselines_configs/${ALGO}/${TASK}/learner seed=${SEED}  # Box2D
```

**Ablations.** `CELL` is `kappa0` (w/o Source-Aware), `norl` (w/o DPG), or `argmax` (w/o Sampling).

```bash
python -m scripts.train_OPI --config-name metaworld/OPI_${TASK}_sparse_${CELL} seed=${SEED}    # assembly, drawer-open, hammer, push-wall
python -m scripts.train_OPI --config-name dmc/OPI_${TASK}_${CELL} seed=${SEED}                # hopper, quad
python -m scripts.train_OPI_simba --config-name box2d/OPI_${TASK}_learner_${CELL} seed=${SEED} # bipedal, weather

# w/o GABC, then Scheduled Exploration (MetaWorld shown; for Quadruped-Run use
# dmc/OPI_quad, drop env.sparse_reward and training.capacity, and drop the '+' prefixes)
python -m scripts.train_OPI --config-name metaworld/OPI_${TASK} training.l1_loss_weight=0.0 \
    env.sparse_reward=true training.total_timesteps=500000 training.capacity=500000 seed=${SEED}
python -m scripts.train_OPI --config-name metaworld/OPI_${TASK} +training.scheduled_exploration=true \
    +training.scheduled_exploration_prob=0.8 +training.scheduled_exploration_steps=500000 \
    env.sparse_reward=true training.total_timesteps=500000 training.capacity=500000 seed=${SEED}
```

## Citation

```bibtex
@inproceedings{ji2026compose,
  title     = {Compose Your Oracles: Off Policy Improvement with Aggregated Guidance},
  author    = {Ji, Jingtian and Liu, Xuefeng and Walter, Matthew R.},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT; see `LICENSE`. `third_party/` contains code adapted from Gymnasium and Meta-World (MIT); see the `NOTICE` files there.
