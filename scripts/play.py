import argparse
import os

from cpg_rl_cbo.isaaclab import cli_args
from cpg_rl_cbo.isaaclab.availability import require_module, require_isaaclab


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained IsaacLab policy.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
    parser.add_argument("--video_length", type=int, default=2000, help="Length of recorded video (steps).")
    parser.add_argument("--enable_cpg", action="store_true", default=False, help="Enable CPG-based policy heads.")
    parser.add_argument("--history_length", default=0, type=int, help="Length of history buffer.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print resolved config and exit.")
    cli_args.add_task_args(parser)
    cli_args.add_runner_args(parser)
    args = parser.parse_args()

    task_spec = cli_args.parse_task_spec(args)
    if args.dry_run:
        print(cli_args.describe_task_spec(task_spec))
        return

    require_isaaclab()
    AppLauncher = require_module("omni.isaac.lab.app", "Install IsaacLab app launcher.").AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    task_spec = cli_args.parse_task_spec(args)
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    gym = require_module("gymnasium", "Install gymnasium for IsaacLab environments.")
    torch = require_module("torch", "Install PyTorch for RSL-RL training.")
    runner_mod = require_module("rsl_rl.runners", "Install rsl_rl for IsaacLab training.")
    lab_tasks = require_module("omni.isaac.lab_tasks.utils", "Install IsaacLab tasks utilities.")
    wrappers = require_module(
        "omni.isaac.lab_tasks.utils.wrappers.rsl_rl",
        "Install IsaacLab RSL-RL wrappers.",
    )

    env_cfg = lab_tasks.parse_env_cfg(task_spec.isaaclab_task, num_envs=task_spec.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_spec, args, play=True)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root_path, args.load_run)

    log_agent_cfg_path = os.path.join(log_dir, "params", "agent.yaml")
    if os.path.exists(log_agent_cfg_path):
        log_agent_cfg_dict = lab_tasks.load_yaml(log_agent_cfg_path)
        lab_tasks.update_class_from_dict(agent_cfg, log_agent_cfg_dict)

    env = gym.make(task_spec.isaaclab_task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.history_length > 0:
        env = wrappers.RslRlVecEnvHistoryWrapper(env, history_length=args.history_length)
    else:
        env = wrappers.RslRlVecEnvWrapper(env)

    resume_path = lab_tasks.get_checkpoint_path(log_root_path, args.load_run, agent_cfg.load_checkpoint)
    runner = runner_mod.OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
