import argparse
from argparse import Namespace

from environment import ArmEnv
from neural_cbf.systems import ArmLidar
from neural_cbf.datamodules.episodic_datamodule import EpisodicDataModule


def build_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Generate ArmLidar dataset only (no training)")

    # robot / environment
    parser.add_argument("--robot_name", type=str, default="panda")
    parser.add_argument("--dis_threshold", type=float, default=0.05)
    parser.add_argument("--simulation_dt", type=float, default=1 / 120)
    parser.add_argument("--controller_period", type=float, default=1 / 30)

    # observation
    parser.add_argument("--observation_type", type=str, default="uniform_lidar")
    parser.add_argument("--point_dim", type=int, default=3)
    parser.add_argument("--n_observation", type=int, default=256)
    parser.add_argument("--n_observation_dataset", type=int, default=512)
    # Keep point-velocity off by default; we now record obstacle joint velocities in aux.
    parser.add_argument("--include_point_velocity", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="ocbf_panda_vel")

    # datamodule
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--noise_level", type=float, default=0.3)
    parser.add_argument("--safe_portion", type=float, default=0.8)
    parser.add_argument("--unsafe_portion", type=float, default=0.2)
    parser.add_argument("--goal_portion", type=float, default=0.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_episode", type=int, default=100)
    parser.add_argument("--trajectories_per_episode", type=int, default=40)
    parser.add_argument("--trajectory_length", type=int, default=35)
    parser.add_argument("--fixed_samples", type=int, default=400)
    parser.add_argument("--skip_fixed_sampling", action="store_true")

    # obstacle robot
    parser.add_argument("--obstacle_robot_name", type=str, default="panda")
    parser.add_argument("--obstacle_traj_path", type=str, default="data/obstacle_trajs/panda_trajs.npz")
    parser.add_argument("--obstacle_horizon_s", type=float, default=0.1)
    parser.add_argument("--obstacle_block_dist", type=float, default=0.1)
    parser.add_argument("--obstacle_block_check_steps", type=int, default=20)

    return parser.parse_args()


def main() -> None:
    args = build_args()

    environment = ArmEnv(
        [args.robot_name],
        GUI=False,
        config_file="",
        obstacle_robot_name=args.obstacle_robot_name,
        obstacle_traj_path=args.obstacle_traj_path,
    )
    robot = environment.robot_list[0]

    dynamics_model = ArmLidar(
        {},
        dis_threshold=args.dis_threshold,
        dt=args.simulation_dt,
        controller_dt=args.controller_period,
        n_obs=args.n_observation,
        point_dim=args.point_dim,
        add_normal=True,
        include_point_velocity=False,
        point_in_dataset_pc=args.n_observation_dataset,
        list_sensor=robot.body_joints,
        env=environment,
        robot=robot,
        observation_type=args.observation_type,
        obstacle_horizon_s=args.obstacle_horizon_s,
    )

    initial_conditions = [tuple(robot.body_range[i]) for i in range(robot.body_dim)]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        total_point=args.n_observation_dataset,
        max_episode=args.max_episode,
        trajectories_per_episode=args.trajectories_per_episode,
        trajectory_length=args.trajectory_length,
        fixed_samples=args.fixed_samples,
        val_split=args.val_split,
        batch_size=args.batch_size,
        noise_level=args.noise_level,
        quotas={
            "safe": args.safe_portion,
            "goal": args.goal_portion,
            "unsafe": args.unsafe_portion,
        },
        name=args.dataset_name,
        obstacle_block_dist=args.obstacle_block_dist,
        obstacle_block_check_steps=args.obstacle_block_check_steps,
        skip_fixed_sampling=args.skip_fixed_sampling,
    )

    data_module.prepare_data()
    print("Dataset generated.")


if __name__ == "__main__":
    main()
