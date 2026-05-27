from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Iterable

from ..action_modes import ActionMode
from ..morphology import Terrain
from ..observations import ObservationProfile
from .task_registry import TaskSpec, get_task_spec

DEFAULT_DEPTH_SHAPE = (24, 32)


def add_task_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=str, default="multi_morph_train", help="Task name from registry.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override number of IsaacLab environments.")
    parser.add_argument(
        "--action_mode",
        type=str,
        default=None,
        choices=[mode.value for mode in ActionMode],
        help="Override RL action decoding mode.",
    )
    parser.add_argument(
        "--observation_profile",
        type=str,
        default=None,
        choices=[profile.value for profile in ObservationProfile],
        help="Override observation profile.",
    )
    parser.add_argument(
        "--morphologies",
        type=str,
        default=None,
        help="Comma-separated morphologies for curriculum (e.g. go1,go2,a1,anymal-b).",
    )
    parser.add_argument(
        "--terrains",
        type=str,
        default=None,
        help="Comma-separated terrains (flat,little_rough,slope).",
    )
    parser.add_argument("--cbo_h", type=float, default=None, help="Override CBO height parameter.")


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    arg_group.add_argument("--experiment_name", type=str, default=None, help="Experiment folder name.")
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    arg_group.add_argument("--save_interval", type=int, default=None, help="Checkpoint save interval.")
    arg_group.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from checkpoint.",
    )
    arg_group.add_argument("--load_run", type=str, default=None, help="Run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger backend to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Logging project name (wandb/neptune).",
    )
    arg_group.add_argument("--use_cnn", action="store_true", default=False, help="Use CNN-based policy.")
    arg_group.add_argument("--use_rnn", action="store_true", default=False, help="Use RNN-based policy.")


def parse_task_spec(args: argparse.Namespace) -> TaskSpec:
    spec = get_task_spec(args.task)
    morphologies = _parse_list(args.morphologies, spec.morphologies)
    terrains = _parse_terrains(args.terrains, spec.terrains)
    action_mode = ActionMode(args.action_mode) if args.action_mode else spec.action_mode
    observation_profile = (
        ObservationProfile(args.observation_profile) if args.observation_profile else spec.observation_profile
    )
    cbo_parameters = spec.cbo_parameters
    if args.cbo_h is not None:
        cbo_parameters = replace(cbo_parameters, h=args.cbo_h)
    num_envs = args.num_envs if args.num_envs is not None else spec.num_envs
    return replace(
        spec,
        morphologies=morphologies,
        terrains=terrains,
        action_mode=action_mode,
        observation_profile=observation_profile,
        cbo_parameters=cbo_parameters,
        num_envs=num_envs,
    )


def parse_rsl_rl_cfg(task_spec: TaskSpec, args: argparse.Namespace, play: bool = False):
    from .availability import require_module

    load_cfg_from_registry = require_module(
        "omni.isaac.lab_tasks.utils.parse_cfg",
        "Install IsaacLab tasks registry for RSL-RL configs.",
    ).load_cfg_from_registry

    rslrl_cfg = load_cfg_from_registry(task_spec.isaaclab_task, "rsl_rl_cfg_entry_point")
    if args.seed is not None:
        rslrl_cfg.seed = args.seed
    if args.resume is not None:
        rslrl_cfg.resume = args.resume
    if args.load_run is not None:
        rslrl_cfg.load_run = args.load_run
    if args.checkpoint is not None:
        rslrl_cfg.load_checkpoint = args.checkpoint
    if args.save_interval is not None:
        rslrl_cfg.save_interval = args.save_interval
    if args.run_name is not None:
        rslrl_cfg.run_name = args.run_name
    if args.experiment_name is not None:
        rslrl_cfg.experiment_name = args.experiment_name
    if args.logger is not None:
        rslrl_cfg.logger = args.logger
    if rslrl_cfg.logger in {"wandb", "neptune"} and args.log_project_name:
        rslrl_cfg.wandb_project = args.log_project_name
        rslrl_cfg.neptune_project = args.log_project_name
    if args.use_cnn:
        rslrl_cfg.use_cnn = args.use_cnn
        rslrl_cfg.policy.class_name = "ActorCriticDepthCNN"
        rslrl_cfg.policy.obs_depth_shape = DEFAULT_DEPTH_SHAPE
    if args.use_rnn:
        rslrl_cfg.policy.rnn_input_size = 2 * rslrl_cfg.policy.actor_hidden_dims[-1]
        rslrl_cfg.policy.rnn_hidden_size = 2 * rslrl_cfg.policy.actor_hidden_dims[-1]
        rslrl_cfg.policy.class_name = "ActorCriticDepthCNNRecurrent" if args.use_cnn else "ActorCriticRecurrent"
    if getattr(args, "enable_cpg", False):
        rslrl_cfg.policy.class_name = "ActorCriticCPG"
        rslrl_cfg.policy.enable_cpg = True
        if not hasattr(rslrl_cfg.policy, "cpg_config"):
            rslrl_cfg.policy.cpg_config = {}
        rslrl_cfg.policy.cpg_config.update(
            {
                "action_mode": task_spec.action_mode.value,
                "observation_profile": task_spec.observation_profile.value,
                "cbo_h": task_spec.cbo_parameters.h,
                "morphologies": task_spec.morphologies,
                "terrains": tuple(t.value for t in task_spec.terrains),
            }
        )
    if hasattr(args, "history_length"):
        rslrl_cfg.policy.history_length = args.history_length
    if play:
        rslrl_cfg.resume = True
    return rslrl_cfg


def describe_task_spec(spec: TaskSpec) -> str:
    terrains = ", ".join(t.value for t in spec.terrains)
    morphologies = ", ".join(spec.morphologies)
    return (
        f"Task: {spec.name}\n"
        f"IsaacLab task: {spec.isaaclab_task}\n"
        f"Description: {spec.description}\n"
        f"Morphologies: {morphologies}\n"
        f"Terrains: {terrains}\n"
        f"Action mode: {spec.action_mode.value}\n"
        f"Observation profile: {spec.observation_profile.value}\n"
        f"CBO h: {spec.cbo_parameters.h}\n"
        f"Num envs: {spec.num_envs}"
    )


def _parse_list(value: str | None, default: Iterable[str]) -> tuple[str, ...]:
    if not value:
        return tuple(default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_terrains(value: str | None, default: Iterable[Terrain]) -> tuple[Terrain, ...]:
    if not value:
        return tuple(default)
    items = []
    for entry in value.split(","):
        cleaned = entry.strip()
        if not cleaned:
            continue
        items.append(Terrain(cleaned))
    return tuple(items)
