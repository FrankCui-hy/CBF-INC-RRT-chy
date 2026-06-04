import os
import sys
import inspect

from argparse import ArgumentParser
from importlib_metadata import requires

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import pybullet as p

from environment import ArmEnv

from neural_cbf.controllers import NeuralLidarCBFController
from neural_cbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_cbf.systems import ArmLidar
from neural_cbf.experiments import (
    ExperimentSuite,
    BFContourExperiment,
    LidarRolloutExperiment
)
from neural_cbf.training.utils import current_git_hash

torch.multiprocessing.set_sharing_strategy("file_system")


def normalize_method_flags(args):
    if getattr(args, "cbf_obs_mode", "legacy_oracle") in ("gphi", "raylink_oracle") and getattr(args, "baseline", None) is None:
        args.baseline = False
    if getattr(args, "baseline", None) is None:
        args.baseline = True

    if args.baseline:
        if getattr(args, "obs_backend", None) is None:
            args.obs_backend = "raw"
        if getattr(args, "train_use_fd", None) is None:
            args.train_use_fd = True
    else:
        if getattr(args, "obs_backend", None) is None:
            args.obs_backend = "gphi" if getattr(args, "cbf_obs_mode", "legacy_oracle") == "gphi" else "raw"
        if getattr(args, "train_use_fd", None) is None:
            args.train_use_fd = False if getattr(args, "cbf_obs_mode", "legacy_oracle") == "gphi" else True
    return args


def main(args):
    args = normalize_method_flags(args)
    if args.cbf_obs_mode == "gphi":
        if not args.gphi_ckpt:
            raise ValueError("--cbf_obs_mode gphi requires --gphi_ckpt.")
        if bool('norm' in args.dataset_name):
            raise ValueError("--cbf_obs_mode gphi is point-only. Use a dataset_name without 'norm'.")
        if args.point_dim != 3:
            raise ValueError("--cbf_obs_mode gphi requires --point_dim 3.")
        if args.train_use_fd:
            raise ValueError("--cbf_obs_mode gphi requires --no_train_use_fd.")
    if args.cbf_obs_mode == "raylink_oracle":
        if not args.gphi_ckpt:
            raise ValueError("--cbf_obs_mode raylink_oracle requires --gphi_ckpt for RayLink metadata.")
        if bool('norm' in args.dataset_name):
            raise ValueError("--cbf_obs_mode raylink_oracle is point-only. Use a dataset_name without 'norm'.")
        if args.point_dim != 3:
            raise ValueError("--cbf_obs_mode raylink_oracle requires --point_dim 3.")
        if not args.train_use_fd:
            raise ValueError(
                "cbf_obs_mode='raylink_oracle' currently supports train_use_fd=True only. "
                "Oracle raycast is not differentiable, so analytic chain rule is not supported."
            )
    if args.state_label_cache and args.cbf_obs_mode == "legacy_oracle":
        raise ValueError("--state_label_cache discards oracle observations and cannot be used with legacy_oracle mode.")
    if args.gphi_include_qobs_dynamics:
        if args.cbf_obs_mode != "gphi":
            raise ValueError("--gphi_include_qobs_dynamics is only supported with --cbf_obs_mode gphi.")
        if args.train_use_fd:
            raise ValueError("--gphi_include_qobs_dynamics requires --no_train_use_fd.")

    # Define the scenarios
    nominal_params = {}
    scenarios = [
        nominal_params,
    ]

    # Define environment and agent
    environment = ArmEnv(
        [args.robot_name],
        GUI=False,
        config_file='',
        obstacle_robot_name=args.obstacle_robot_name,
        obstacle_traj_path=args.obstacle_traj_path,
        obstacle_robot_base_pos=(float(args.obst_base_x), float(args.obst_base_y), float(args.obst_base_z)),
        obstacle_robot_base_orn=(0.0, 0.0, 1.0, 0.0),
    )
    robot = environment.robot_list[0]
    # Align training sampling geometry with eval defaults.
    try:
        bpos, born = environment.p.getBasePositionAndOrientation(robot.robotId)
        environment.p.resetBasePositionAndOrientation(
            robot.robotId,
            [float(args.main_base_x), float(args.main_base_y), float(args.main_base_z)],
            born,
        )
    except Exception:
        pass

    # Define the dynamics model
    dynamics_model = ArmLidar(
        nominal_params,
        dis_threshold=args.dis_threshold,
        dt=args.simulation_dt,
        controller_dt=args.controller_period,
        n_obs=args.n_observation,
        point_dim=args.point_dim,
        add_normal=bool('norm' in args.dataset_name),
        include_point_velocity=False,
        point_in_dataset_pc = args.n_observation_dataset,
        list_sensor=robot.body_joints,
        env=environment,
        robot=robot,
        observation_type=args.observation_type,
        obstacle_horizon_s=args.obstacle_horizon_s,
    )

    # Define goal_state for validation
    goal_state = torch.tensor(robot.q0).float()
    dynamics_model.set_goal(goal_state)

    # Initialize the DataModule
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
        quotas={"safe": args.safe_portion, "goal": args.goal_portion, "unsafe": args.unsafe_portion},
        name=args.dataset_name,
        obstacle_block_dist=args.obstacle_block_dist,
        obstacle_block_check_steps=args.obstacle_block_check_steps,
        state_label_cache=args.state_label_cache,
    )


    # Define the experiment suite
    exp_suite_list = []

    if args.exp_cbf_contour:
        default_state = dynamics_model.complete_sample_with_observations(dynamics_model.goal_state.reshape(1, -1), num_samples=1).squeeze()
        cbf_contour_experiment = BFContourExperiment(
            "cbf_Contour",
            domain=[tuple(robot.body_range[args.contour_x_idx]), tuple(robot.body_range[args.contour_y_idx])],
            n_grid=30,
            x_axis_index=args.contour_x_idx,
            y_axis_index=args.contour_y_idx,
            x_axis_label=f"$\\theta_{args.contour_x_idx}$",
            y_axis_label=f"$\\theta_{args.contour_y_idx}$",
            default_state=default_state,
            plot_unsafe_region=True,
        )
        exp_suite_list.append(cbf_contour_experiment)

    if args.exp_rollout:
        ul, ll = dynamics_model.state_limits
        start_x = torch.cat([
            torch.lerp(ll, ul, 0.2 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
            torch.lerp(ll, ul, 0.8 * torch.ones(ll.shape[-1]).double()).reshape(1, -1),
        ], dim=0).float()
        start_x = dynamics_model.complete_sample_with_observations(start_x, num_samples=start_x.shape[0])

        rollout_experiment = LidarRolloutExperiment(
            "Rollout",
            start_x,
            args.rollout_x_idx,
            f"$\\theta_{args.rollout_x_idx}$",
            args.rollout_y_idx,
            f"$\\theta_{args.rollout_y_idx}$",
            scenarios=scenarios,
            n_sims_per_start=args.rollout_n_sim_per_start,
            t_sim=args.rollout_t_sim,
        )
        exp_suite_list.append(rollout_experiment)

    experiment_suite = ExperimentSuite(exp_suite_list)

    # Initialize the controller
    loss_config = {
        "u_coef_in_training": args.u_coef_in_training,
        "safe_classification_weight": args.safe_classification_weight,
        "unsafe_classification_weight": args.unsafe_classification_weight,
        "descent_violation_weight": args.descent_violation_weight,
        "hdot_divergence_weight": args.hdot_divergence_weight,
        "epsilon": getattr(args, "epsilon", 0.0),
    }
    requested_version = args.version
    method_tag = "baseline_fd_raw" if args.baseline else f"cbf_obs_{args.cbf_obs_mode}"
    version_name = requested_version if method_tag in requested_version else f"{requested_version}_{method_tag}"
    args.method_tag = method_tag
    args.version = version_name
    cbf_controller = NeuralLidarCBFController(dynamics_model, scenarios, data_module, experiment_suite,
                                              safe_level=args.safe_level,
                                              unsafe_level=args.unsafe_level,
                                              cbf_hidden_layers=args.cbf_hidden_layers,
                                              cbf_hidden_size=args.cbf_hidden_size,
                                              cbf_alpha=args.cbf_alpha,
                                              cbf_relaxation_penalty=5000,
                                              feature_dim=args.feature_dim,
                                              per_feature_dim=args.per_feature_dim,
                                              learn_shape_epochs=args.learn_shape_epochs,
                                              loss_config=loss_config,
                                              all_hparams=args,
                                              use_bn=args.use_bn,
                                              ab_mode=args.ab_mode,
                                              baseline=args.baseline,
                                              obs_backend=args.obs_backend,
                                              cbf_obs_mode=args.cbf_obs_mode,
                                              gphi_ckpt=args.gphi_ckpt,
                                              gphi_hit_threshold=args.gphi_hit_threshold,
                                              gphi_hit_temp=args.gphi_hit_temp,
                                              gphi_freeze=args.gphi_freeze,
                                              gphi_include_qobs_dynamics=args.gphi_include_qobs_dynamics,
                                              train_use_fd=args.train_use_fd,
                                              use_neural_actor="RL" in requested_version,)

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.abspath(__file__).rsplit('/', 3)[0] + f"/models/neural_cbf/{dynamics_model}",
        name=f"{args.version}", #_gpu{args.devices}",
    )
    trainer_kwargs = {
        "logger": tb_logger,
        "reload_dataloaders_every_epoch": True,
        "max_epochs": args.max_epochs,
    }
    if args.ckpt and "resume_from_checkpoint" in inspect.signature(pl.Trainer.__init__).parameters:
        trainer_kwargs["resume_from_checkpoint"] = args.ckpt

    if torch.cuda.is_available():
        trainer = pl.Trainer(
            gpus=args.devices,  # only supporting single-GPU at present
            **trainer_kwargs,
        )
    else:
        trainer = pl.Trainer(**trainer_kwargs)

    # Train
    pl.seed_everything(args.seed)
    torch.autograd.set_detect_anomaly(False)
    if args.ckpt:
        # Torch>=2.6 defaults torch.load(weights_only=True), which breaks older
        # Lightning checkpoint restore paths that expect full object unpickling.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    fit_kwargs = {}
    if args.ckpt and "ckpt_path" in inspect.signature(pl.Trainer.fit).parameters:
        fit_kwargs["ckpt_path"] = args.ckpt
    trainer.fit(cbf_controller, **fit_kwargs)


if __name__ == "__main__":
    parser = ArgumentParser()

    # environment params
    parser.add_argument('--robot_name', type=str, default='panda')
    parser.add_argument('--version', type=str, default="multiple_seeds")

    # simulation params
    parser.add_argument('--dis_threshold', type=float, default=0.05)
    parser.add_argument('--controller_period', type=float, default=1/30)
    parser.add_argument('--simulation_dt', type=float, default=1/120)

    # CBF definition params
    parser.add_argument('--safe_level', type=float, default=0.1, help='h_safe < -safe_level')
    parser.add_argument('--unsafe_level', type=float, default=0.1, help='h_unsafe > unsafe_level')

    # training params
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('--devices', type=str, default="1", help='gpu id')
    parser.add_argument('--max_epochs', type=int, default=121)
    parser.add_argument('--learn_shape_epochs', type=int, default=-1,
                        help='different from max_epochs when training a neural policy')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for CBF optimizer')
    parser.add_argument('--ckpt', type=str, default='',
                        help='Path to a Lightning checkpoint to resume training from.')

    # neural network params
    parser.add_argument('--cbf_hidden_layers', type=int, default=2)
    parser.add_argument('--cbf_hidden_size', type=int, default=48)
    parser.add_argument('--cbf_alpha', type=float, default=1, help='lambda in (L_f V + L_g V u + lambda V <= 0)')
    parser.add_argument('--per_feature_dim', type=int, default=64, help='local feature extracted from each point cloud')
    parser.add_argument('--feature_dim', type=int, default=32, help='global feature extracted from encoder')
    parser.add_argument('--use_bn', type=bool, default=False, help='global feature extracted from encoder')

    # loss config params
    parser.add_argument('--u_coef_in_training', type=float, default=5e-1, help='control signal amplification coefficient in training')
    parser.add_argument('--safe_classification_weight', type=float, default=20, help='weight of safe region classification loss')
    parser.add_argument('--unsafe_classification_weight', type=float, default=20, help='weight of unsafe region classification loss')
    parser.add_argument('--descent_violation_weight', type=float, default=2, help='weight of descent violation loss')
    parser.add_argument('--hdot_divergence_weight', type=float, default=2e-2, help='weight of hdot divergence loss')
    parser.add_argument('--epsilon', type=float, default=0.0, help='CBF residual margin in relu(epsilon + hdot + alpha*h)')

    # observation params
    parser.add_argument('--point_dim', type=int, default=3, help='cartesian or spherical coordinates')
    parser.add_argument('--observation_type', type=str, default='uniform_surface', help='[uniform_lidar, uniform_surface, cone_lidar]')
    parser.add_argument('--n_observation', type=int, default=256, help='num of rays from each lidar sensor')

    # obstacle robot params
    parser.add_argument('--obstacle_robot_name', type=str, default='panda')
    parser.add_argument('--obstacle_traj_path', type=str, default='data/obstacle_trajs/panda_trajs.npz')
    parser.add_argument('--main_base_x', type=float, default=0.0)
    parser.add_argument('--main_base_y', type=float, default=-0.25)
    parser.add_argument('--main_base_z', type=float, default=0.0)
    parser.add_argument('--obst_base_x', type=float, default=0.0)
    parser.add_argument('--obst_base_y', type=float, default=0.25)
    parser.add_argument('--obst_base_z', type=float, default=0.0)
    parser.add_argument('--obstacle_horizon_s', type=float, default=0.2)
    parser.add_argument('--obstacle_block_dist', type=float, default=0.1)
    parser.add_argument('--obstacle_block_check_steps', type=int, default=20)

    # datamodule params
    parser.add_argument('--dataset_name', type=str, default='pino_motor_norm', help='[5dpoints, motor_control]')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_observation_dataset', type=int, default=256, help='total points in ')
    parser.add_argument('--noise_level', type=float, default=0.3)
    parser.add_argument('--safe_portion', type=float, default=0.8, help='portion of safe dps in dataset')
    parser.add_argument('--unsafe_portion', type=float, default=0.2, help='portion of unsafe dps in dataset')
    parser.add_argument('--goal_portion', type=float, default=0.0, help='portion of goal dps in dataset')
    parser.add_argument('--val_split', type=float, default=0.1, help='portion of validation dps in dataset')
    parser.add_argument('--max_episode', type=int, default=120)
    parser.add_argument('--trajectories_per_episode', type=int, default=40)
    parser.add_argument('--trajectory_length', type=int, default=35)
    parser.add_argument('--fixed_samples', type=int, default=200)
    parser.add_argument(
        '--ab_mode',
        type=str,
        default='B_with_normal',
        choices=['A_no_normal', 'B_with_normal'],
        help='A/B switch: A masks normal channels before encoder, B uses full observation.',
    )
    parser.add_argument('--baseline', dest='baseline', action='store_true', default=None,
                        help='Use legacy FD/simulated hdot chain.')
    parser.add_argument('--no_baseline', dest='baseline', action='store_false',
                        help='Use Safe_Dual analytic g_phi chain.')
    parser.add_argument('--obs_backend', type=str, default=None, choices=['gphi', 'raw'],
                        help='Observation backend for training. Defaults to raw for baseline and gphi otherwise.')
    parser.add_argument(
        '--gphi_ckpt',
        type=str,
        default='',
        help='FK-aware RayLink g_phi checkpoint path.',
    )
    parser.add_argument('--cbf_obs_mode', type=str, default='legacy_oracle',
                        choices=['legacy_oracle', 'gphi', 'raylink_oracle'],
                        help='Observation source for CBF training.')
    parser.add_argument('--gphi_hit_threshold', type=float, default=0.5)
    parser.add_argument('--gphi_hit_temp', type=float, default=0.1)
    parser.add_argument('--gphi_freeze', dest='gphi_freeze', action='store_true', default=True)
    parser.add_argument('--no_gphi_freeze', dest='gphi_freeze', action='store_false')
    parser.add_argument('--gphi_include_qobs_dynamics', action='store_true', default=False,
                        help='Include dH/dq_obs @ qdot_obs in the RayLink gphi analytic CBF derivative.')
    parser.add_argument('--state_label_cache', type=str, default='',
                        help='Optional cbf_state_label_v1 cache. Only valid for gphi/raylink_oracle modes.')
    parser.add_argument('--train_use_fd', dest='train_use_fd', action='store_true', default=None,
                        help='Compute FD hdot in training (debug only, slower). Default: disabled.')
    parser.add_argument('--no_train_use_fd', dest='train_use_fd', action='store_false',
                        help='Use analytic/min-control hdot in training.')
    # ## for debugging
    # parser.add_argument('--max_episode', type=int, default=2)
    # parser.add_argument('--trajectories_per_episode', type=int, default=5)
    # parser.add_argument('--trajectory_length', type=int, default=50)
    # parser.add_argument('--fixed_samples', type=int, default=30)

    # experiment-suite params
    # parser.add_argument('--exp_cbf_contour', action='store_true')
    parser.add_argument('--exp_cbf_contour', action='store_false')
    parser.add_argument('--contour_x_idx', type=int, default=1)
    parser.add_argument('--contour_y_idx', type=int, default=3)
    # parser.add_argument('--exp_rollout', action='store_true')
    parser.add_argument('--exp_rollout', action='store_false')
    parser.add_argument('--rollout_x_idx', type=int, default=1)
    parser.add_argument('--rollout_y_idx', type=int, default=3)
    parser.add_argument('--rollout_t_sim', type=float, default=3.)
    parser.add_argument('--rollout_n_sim_per_start', type=int, default=2)

    args = parser.parse_args()



    main(args)
