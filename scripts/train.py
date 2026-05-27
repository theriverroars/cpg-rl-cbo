import argparse
import os
from datetime import datetime

from cpg_rl_cbo.isaaclab import cli_args
from cpg_rl_cbo.isaaclab.availability import require_module, require_isaaclab


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL agent with IsaacLab + RSL-RL.")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (steps).")
    parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (steps).")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL training iterations.")
    parser.add_argument("--enable_cpg", action="store_true", default=True, help="Enable CPG-based policy heads.")
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    gym = require_module("gymnasium", "Install gymnasium for IsaacLab environments.")
    torch = require_module("torch", "Install PyTorch for RSL-RL training.")
    runner_mod = require_module("rsl_rl.runners", "Install rsl_rl for IsaacLab training.")
    lab_tasks = require_module("omni.isaac.lab_tasks.utils", "Install IsaacLab tasks utilities.")
    wrappers = require_module(
        "omni.isaac.lab_tasks.utils.wrappers.rsl_rl",
        "Install IsaacLab RSL-RL wrappers.",
    )

    env_cfg = lab_tasks.parse_env_cfg(task_spec.isaaclab_task, num_envs=task_spec.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_spec, args)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if args.max_iterations:
        agent_cfg.max_iterations = args.max_iterations

    env = gym.make(task_spec.isaaclab_task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos"),
            step_trigger=lambda step: step % args.video_interval == 0,
            video_length=args.video_length,
            disable_logger=False,
        )
    if args.history_length > 0:
        env = wrappers.RslRlVecEnvHistoryWrapper(env, history_length=args.history_length)
    else:
        env = wrappers.RslRlVecEnvWrapper(env)

    agent_cfg_dict = agent_cfg.to_dict()
    runner = runner_mod.OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = lab_tasks.get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner.load(resume_path)

    env.seed(agent_cfg.seed)
    lab_tasks.dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    lab_tasks.dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    lab_tasks.dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    lab_tasks.dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
