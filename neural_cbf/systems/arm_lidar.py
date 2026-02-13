"""Define observation-related dynamics for a 7-DoF Franka Panda robot"""
import time
from typing import Tuple, List, Optional, Callable, Dict

import os
import torch
import numpy as np
import tqdm

from environment import ArmEnv, BasicRobot

from neural_cbf.systems import ArmDynamics
from .utils import grav, cartesian_to_spherical, spherical_to_cartesian
from .utils import Scenario


class ArmLidar(ArmDynamics):
    """
    The observation is a point cloud in world frame.

    State: q + o + aux
    """

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float,
        controller_dt: Optional[float],
        dis_threshold: float,
        env: ArmEnv = None,
        robot: BasicRobot = None,
        # observation-related, more than base dynamics
        n_obs: int = 128,
        point_in_dataset_pc: int = 0,
        list_sensor=None,
        observation_type: str = "",
        point_dim=4,
        add_normal=False,
        include_point_velocity: bool = False,
        obstacle_horizon_s: float = 0.05,
    ):
        """
        Initialization.
        """
        super().__init__(
            nominal_params,
            dt,
            controller_dt,
            dis_threshold=dis_threshold,
            use_linearized_controller=False,
            env=env,
            robot=robot,
        )

        self.compute_linearized_controller(None)
        self.observation_type = observation_type

        self.add_normal = add_normal
        self.include_point_velocity = include_point_velocity
        if self.include_point_velocity and point_dim != 3:
            raise ValueError("When include_point_velocity=True, point_dim must be 3 (cartesian).")
        self.point_dims = point_dim + 3 * int(self.include_point_velocity) + 3 * int(self.add_normal)
        self.obstacle_horizon_s = obstacle_horizon_s
        self.obstacle_qdot_dim = (
            self.robot.body_dim
            if self.env is not None and self.env.obstacle_robot is not None
            else 0
        )

        # initialize sensor
        self.ray_per_sensor = n_obs
        self.point_in_dataset_pc = point_in_dataset_pc

        self.list_sensor = list_sensor

        if self.observation_type == "uniform_lidar":
            self.rayFromLocal = np.zeros((self.point_in_dataset_pc, 3))
            theta = np.random.normal((0, 0, 0), (1, 1, 1), (self.point_in_dataset_pc, 3))
            self.rayToLocal = theta / np.linalg.norm(theta, axis=1, keepdims=True)

    def __str__(self):
        return f"{str(self.robot)}_lidar_Dynamics"

    @property
    def o_dims_in_dataset(self) -> int:
        return self.point_in_dataset_pc * (
            3 + 3 * int(self.include_point_velocity) + 3 * int(self.add_normal)
        )

    @property
    def o_dims(self) -> int:
        return len(self.list_sensor) * self.ray_per_sensor * self.point_dims

    @property
    def state_aux_dims_in_dataset(self) -> int:
        if self.obstacle_qdot_dim > 0:
            return self.sensor_aux_dims + self.obstacle_qdot_dim + 2
        return self.sensor_aux_dims

    @property
    def state_aux_dims(self) -> int:
        return 0

    @property
    def sensor_aux_dims(self) -> int:
        return len(self.list_sensor) * (3 + 9)

    def _infer_observation_from_datax(self, datax: torch.Tensor) -> None:
        """
        Infer per-point observation dimension from datax shape and update flags.
        Expected per-point dims:
          3 -> position
          6 -> position + normal
          9 -> position + velocity + normal
        """
        total = datax.shape[1]
        base = self.n_dims + self.sensor_aux_dims
        extra = total - base
        # remove obstacle meta if present
        if self.obstacle_qdot_dim > 0:
            extra -= (self.obstacle_qdot_dim + 2)
        else:
            # try to guess obstacle meta length (qdot_obs + traj_idx + step_idx)
            guess_meta = self.robot.body_dim + 2
            if extra - guess_meta > 0:
                extra -= guess_meta
        if extra <= 0:
            return
        point_dim_dataset = extra // self.point_in_dataset_pc
        if point_dim_dataset == 3:
            self.include_point_velocity = False
            self.add_normal = False
        elif point_dim_dataset == 6:
            self.include_point_velocity = False
            self.add_normal = True
        elif point_dim_dataset == 9:
            self.include_point_velocity = True
            self.add_normal = True
        else:
            # Unknown layout; do not mutate
            return
        self.point_dims = 3 + 3 * int(self.include_point_velocity) + 3 * int(self.add_normal)

    def _infer_obstacle_qdot_dim_from_datax(self, datax: torch.Tensor) -> None:
        """
        Infer obstacle_qdot_dim from datax shape if not already set.
        datax shape should be: n_dims + o_dims_in_dataset + (sensor_aux + qdot_obs + 2)
        """
        if self.obstacle_qdot_dim > 0:
            return
        total = datax.shape[1]
        base = self.n_dims + self.o_dims_in_dataset + self.sensor_aux_dims
        extra = total - base
        if extra >= 2:
            self.obstacle_qdot_dim = extra - 2

    def _get_observation_with_state(self, state):
        if self.observation_type == "uniform_surface":
            obs = self.env.sample_obstacle_surface(
                self.point_in_dataset_pc, add_normal=self.add_normal
            )
            if self.include_point_velocity:
                zeros_v = np.zeros((obs.shape[0], 3), dtype=obs.dtype)
                obs = np.concatenate((obs[:, :3], zeros_v, obs[:, 3:]), axis=1)
            return obs.reshape(-1)

        elif self.observation_type == "uniform_lidar":
            """
            uniformly ejecting light rays from the sensor on a uniform sphere surface, and check the collision
            """
            obs = np.zeros((self.point_in_dataset_pc, self.point_dims), dtype=np.float32)

            # NOTE: 原代码这里 sensors 只取两个 link（末端 & 中间），保持不动
            sensors = [self.robot.body_joints[-1], self.robot.body_joints[self.robot.body_dim // 2]]
            each_obstacle = np.random.randint(
                low=0, high=len(sensors), size=self.point_in_dataset_pc
            )

            fk = self.robot.forward_kinematics(self.list_sensor, state[: self.q_dims])

            vel_map = None
            if self.include_point_velocity and self.env is not None and self.env.obstacle_robot is not None:
                # Cache link velocities once per observation for speed.
                vel_map = {}
                base_lin_vel, _ = self.env.p.getBaseVelocity(self.env.obstacle_robot.robotId)
                vel_map[-1] = np.array(base_lin_vel, dtype=np.float32)
                for link_id in range(self.env.obstacle_robot.n_joints):
                    link_state = self.env.p.getLinkState(
                        self.env.obstacle_robot.robotId,
                        int(link_id),
                        computeLinkVelocity=1,
                    )
                    vel_map[int(link_id)] = np.array(link_state[6], dtype=np.float32)

            for sensor in sensors:
                origin = fk[sensor][0]
                # orientation = fk[sensor][1]  # 未使用，保留注释即可

                which_point = np.where(each_obstacle == sensors.index(sensor))[0]

                rayFrom = self.rayFromLocal[which_point] + origin
                rayTo = self.rayToLocal[which_point] + origin

                raw_results = self.env.p.rayTestBatch(
                    rayFrom.reshape((-1, 3)),
                    rayTo.reshape((-1, 3)),
                    numThreads=0,  # 想提速可改成 4/8（看 CPU）
                )
                valid_obstacle_ids = self.env.obstacle_ids
                if self.env is not None and self.env.obstacle_robot is not None:
                    valid_obstacle_ids = [self.env.obstacle_robot.robotId]

                pos = np.zeros((rayFrom.shape[0], 3), dtype=np.float32)
                nrm = np.zeros((rayFrom.shape[0], 3), dtype=np.float32)
                link_ids = np.full((rayFrom.shape[0],), -1, dtype=np.int32)

                for k, r in enumerate(raw_results):
                    hit_uid = r[0]
                    hit_link = r[1]
                    hit_pos = r[3]
                    hit_nrm = r[4]

                    if hit_uid in valid_obstacle_ids:
                        pos[k, :] = hit_pos
                        link_ids[k] = hit_link
                        if self.add_normal:
                            nrm[k, :] = hit_nrm
                    else:
                        pos[k, :] = rayTo[k, :]

                refined_results = pos
                if self.include_point_velocity:
                    vel = np.zeros((pos.shape[0], 3), dtype=pos.dtype)
                    if vel_map is not None:
                        for k, link_id in enumerate(link_ids):
                            vel[k, :] = vel_map.get(int(link_id), np.zeros((3,), dtype=pos.dtype))
                    refined_results = np.concatenate((refined_results, vel), axis=1)
                if self.add_normal:
                    refined_results = np.concatenate((refined_results, nrm), axis=1)

                # 现在 refined_results 的行数 == which_point.shape[0]，列数 == point_dims
                obs[which_point, :] = refined_results

            return obs.reshape(-1)

        else:
            raise NotImplementedError(f"Unknown observation type: {self.observation_type}")

    def complete_sample_with_observations(
        self, x, num_samples: int, obstacle_steps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        input: bs * (n_dim + o_dim + aux_dim)
        """
        samples = torch.zeros(
            num_samples,
            self.n_dims + self.o_dims_in_dataset + self.state_aux_dims_in_dataset,
            device=x.device,
        )
        samples[:, : self.n_dims] = x
        for i in range(num_samples):
            if self.env is not None and self.env.obstacle_robot is not None:
                if obstacle_steps is not None:
                    self.env.apply_obstacle_step(int(obstacle_steps[i].item()))
            o = self._get_observation_with_state(x[i, : self.q_dims])
            samples[i, self.n_dims : -self.state_aux_dims_in_dataset] = torch.tensor(
                o, device=x.device
            )
            samples[i, -self.state_aux_dims_in_dataset :] = self.get_aux(x[i, : self.q_dims])
        return samples

    def get_aux(self, state):
        state_aux = []
        x_fk = self.robot.forward_kinematics(self.list_sensor, state[: self.q_dims])
        for i in range(len(x_fk)):
            p_p = torch.tensor(x_fk[i][0], device=state.device)
            p_r = torch.tensor(x_fk[i][1], device=state.device)
            state_aux.append(torch.cat((p_p.reshape(1, -1), p_r.reshape(1, -1)), dim=1))
        aux = torch.cat(state_aux, dim=0).reshape(-1)
        if self.env is not None and self.env.obstacle_robot is not None:
            qdot_obs = self.env.get_obstacle_qdot()
            traj_idx = float(self.env.obstacle_traj_idx)
            step_idx = float(self.env.obstacle_traj_step)
            aux = torch.cat(
                (
                    aux,
                    torch.tensor(qdot_obs, device=state.device, dtype=aux.dtype),
                    torch.tensor([traj_idx, step_idx], device=state.device, dtype=aux.dtype),
                ),
                dim=0,
            )
        return aux

    def get_jacobian(self, state):
        J_p = torch.zeros((len(self.list_sensor), 3, self.q_dims))
        J_R = torch.zeros((len(self.list_sensor), 3, 3, self.q_dims))
        for a_idx, sensor_link in zip(range(len(self.list_sensor)), self.list_sensor):
            J_p[a_idx, :, :] = torch.from_numpy(
                self.robot.get_jacobian(state.tolist(), sensor_link, [0, 0, 0])[0]
            )
            J_W = self.robot.get_jacobian(state.tolist(), sensor_link, [0, 0, 0])[1]
            # dW_dq to dR_dq
            aux = self.get_aux(state)
            transformation = aux[: self.sensor_aux_dims].reshape(-1, 12).float()
            R_matrix = transformation[a_idx, 3:].reshape(3, 3)
            J_R[a_idx, 0, 1, :] = torch.from_numpy(-J_W[2, :])
            J_R[a_idx, 0, 2, :] = torch.from_numpy(J_W[1, :])
            J_R[a_idx, 1, 0, :] = torch.from_numpy(J_W[2, :])
            J_R[a_idx, 1, 2, :] = torch.from_numpy(-J_W[0, :])
            J_R[a_idx, 2, 0, :] = torch.from_numpy(-J_W[1, :])
            J_R[a_idx, 2, 1, :] = torch.from_numpy(J_W[0, :])
            J_R[a_idx, :, :, :] = torch.einsum("ibk,bj->ijk", J_R[a_idx, :, :, :], R_matrix)
        return J_p.unsqueeze(0), J_R.unsqueeze(0)

    def get_batch_jacobian(self, x):
        Js_P = []
        Js_R = []
        for idx in range(x.shape[0]):
            J_P, J_R = self.get_jacobian(x[idx, : self.n_dims])
            Js_P.append(J_P)
            Js_R.append(J_R)
        Js_P = torch.cat(Js_P, dim=0)
        Js_R = torch.cat(Js_R, dim=0)
        return Js_P, Js_R

    def datax_to_x(self, x: torch.Tensor):
        # x: bs * (n_dim + o_dim_in_dataset + aux_dim_in_dataset)
        self._infer_observation_from_datax(x)
        bs = x.shape[0]
        q = x[:, : self.n_dims]
        point_dim_dataset = 3 + 3 * int(self.include_point_velocity) + 3 * int(self.add_normal)
        global_obs = x[:, self.n_dims : -self.state_aux_dims_in_dataset].reshape(
            bs, -1, point_dim_dataset
        )
        transformation = (
            x[:, -self.state_aux_dims_in_dataset :][:, : self.sensor_aux_dims]
            .reshape(bs, -1, 12)
        )

        origins = transformation[:, :, :3]
        rotation_matrixs = transformation[:, :, 3:].reshape(bs, -1, 3, 3)

        obs = torch.zeros(
            (bs, len(self.list_sensor), self.ray_per_sensor, self.point_dims), device=x.device
        )
        for idx in range(len(self.list_sensor)):
            origin = origins[:, idx, :]
            rotation_matrix = rotation_matrixs[:, idx, :, :]

            sampled_index = torch.randint(
                low=0, high=global_obs.shape[1], size=(self.ray_per_sensor, 1), device=x.device
            ).squeeze().int()
            raw_results = torch.index_select(global_obs, dim=1, index=sampled_index)
            pos = raw_results[:, :, :3]
            offset_pos = torch.transpose(
                torch.bmm(torch.transpose(rotation_matrix, 1, 2), torch.transpose(pos - origin.unsqueeze(1), 1, 2)),
                1,
                2,
            )
            if self.point_dims == 4 and not self.include_point_velocity:
                offset_pos = cartesian_to_spherical(offset_pos.reshape(-1, 3)).reshape(bs, -1, 4)
            parts = [offset_pos]
            cursor = 3
            if self.include_point_velocity:
                vel = raw_results[:, :, cursor : cursor + 3]
                vel_local = torch.transpose(
                    torch.bmm(torch.transpose(rotation_matrix, 1, 2), torch.transpose(vel, 1, 2)),
                    1,
                    2,
                )
                parts.append(vel_local)
                cursor += 3
            if self.add_normal:
                norm = raw_results[:, :, cursor : cursor + 3]
                norm_local = torch.transpose(
                    torch.bmm(torch.transpose(rotation_matrix, 1, 2), torch.transpose(norm, 1, 2)),
                    1,
                    2,
                )
                parts.append(norm_local)
            refined_results = torch.cat(parts, dim=2)
            obs[:, idx, :, :] = refined_results
        return torch.cat((q, obs.reshape(bs, -1)), dim=1)

    def get_obstacle_meta_from_datax(self, datax: torch.Tensor):
        self._infer_observation_from_datax(datax)
        self._infer_obstacle_qdot_dim_from_datax(datax)
        if self.obstacle_qdot_dim == 0:
            return None, None, None
        meta = datax[:, -self.state_aux_dims_in_dataset :]
        qdot = meta[:, -(self.obstacle_qdot_dim + 2) : -2]
        traj_idx = meta[:, -2]
        step_idx = meta[:, -1]
        return qdot, traj_idx, step_idx

    def batch_lookahead(self, datax, dqs, data_jacobian):
        """
        estimate next-step observation, x: state + observation, dqs: a list of possible dq
        (now estimation/learning for future work especially in dynamic env)
        """
        if self.point_dims in [3, 4, 6, 9]:
            datax_next = datax.detach().clone()
            datax_next[:, : self.n_dims] += dqs
            if len(data_jacobian) > 0:
                aux = datax[:, -self.state_aux_dims_in_dataset :]
                sensor_aux = aux[:, : self.sensor_aux_dims]
                meta_aux = aux[:, self.sensor_aux_dims :]
                sensor_aux_next = self._jacobian_to_auxlookahead(sensor_aux, data_jacobian, dqs)
                datax_next[:, -self.state_aux_dims_in_dataset :] = torch.cat(
                    (sensor_aux_next, meta_aux), dim=1
                )
            else:
                for idx in range(datax.shape[0]):
                    datax_next[idx, -self.state_aux_dims_in_dataset :] = self.get_aux(
                        datax_next[idx, : self.n_dims]
                    )
            return datax_next
        else:
            raise NotImplementedError(f"Unknown point dimension {self.point_dims}.")

    def _jacobian_to_auxlookahead(self, aux, jacobian, dq):
        """
        batch-wise, jacobian: (J_p, J_R), dq: bs * q_dim
        return auxillary dimensions in state, defined by list_sensor * (p + R)
        """
        bs = aux.shape[0]
        q_dim = dq.shape[1]
        transformation = aux.reshape(bs, -1, 12)
        ps = transformation[:, :, :3]
        Rs = transformation[:, :, 3:].reshape(bs, -1, 3, 3)

        dq = dq.unsqueeze(1).expand(-1, transformation.shape[1], -1).reshape(-1, q_dim, 1)

        p_next = ps + torch.bmm(jacobian[0].reshape(dq.shape[0], 3, q_dim), dq).reshape(*ps.shape)
        R_next = Rs + torch.bmm(jacobian[1].reshape(dq.shape[0], 9, q_dim), dq).reshape(*Rs.shape)
        R_next = R_next.reshape(-1, 3, 3)
        aux_next = torch.cat((p_next, R_next.reshape(bs, -1, 9)), dim=2).reshape(bs, -1)
        return aux_next

    def closed_loop_dynamics(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        collect_dataset: bool = False,
        use_motor_control: bool = False,
        update_observation: bool = True,
        return_time: bool = False,
    ) -> torch.Tensor:
        """
        This returns x_next (not xdot) due to existence of observation.
        """
        if return_time:
            ttt_list = [0, 0]  # [offline, online]
            t0 = time.time()

        batch_size = x.shape[0]
        x_next = torch.zeros(
            (batch_size, self.n_dims + self.o_dims_in_dataset + self.state_aux_dims_in_dataset),
            device=x.device,
        )

        step = 10 if collect_dataset else 1
        if return_time:
            ttt_list[0] += time.time() - t0
            t0 = time.time()

        for i in range(batch_size):
            q_dot = u[i]
            xdot = q_dot.type_as(x)

            if use_motor_control:
                self.robot.set_joint_position(self.robot.body_joints, x[i, : self.n_dims])
                self.env.p.setJointMotorControlArray(
                    self.robot.robotId,
                    self.robot.body_joints,
                    self.env.p.VELOCITY_CONTROL,
                    targetVelocities=xdot,
                )
                for _ in range(step):
                    self.env.p.stepSimulation()
                x_next[i, : self.n_dims] = torch.tensor(
                    self.robot.get_joint_position(self.robot.body_joints), device=x.device
                )
            else:
                x_next[i, : self.n_dims] = x[i, : self.n_dims] + xdot * self.dt * step
                self.robot.set_joint_position(self.robot.body_joints, x_next[i, : self.n_dims])

            if return_time:
                ttt_list[0] += time.time() - t0
                t0 = time.time()

            if self.env is not None and self.env.obstacle_robot is not None:
                self.env.step_obstacle(step)

            # observation
            if update_observation:
                o = torch.tensor(self._get_observation_with_state(x_next[i, : self.n_dims]), device=x.device)
            else:
                o = x[i, self.n_dims : -self.state_aux_dims_in_dataset].clone()

            x_next[i, self.n_dims : -self.state_aux_dims_in_dataset] = o
            x_next[i, -self.state_aux_dims_in_dataset :] = self.get_aux(x_next[i, : self.q_dims])

            if return_time:
                ttt_list[1] += time.time() - t0
                t0 = time.time()

        if return_time:
            return x_next, ttt_list
        else:
            return x_next

    def safe_mask(self, x):
        if (
            self.env is None
            or self.env.obstacle_robot is None
            or self.obstacle_qdot_dim == 0
            or self.env.obstacle_traj_dt is None
        ):
            return super().safe_mask(x)

        safe_mask = torch.ones_like(x[:, 0]).type_as(x).to(dtype=torch.bool)
        if x.shape[1] <= self.n_dims:
            traj_idx = torch.full((x.shape[0],), float(self.env.obstacle_traj_idx), device=x.device)
            step_idx = torch.full((x.shape[0],), float(self.env.obstacle_traj_step), device=x.device)
        else:
            _, traj_idx, step_idx = self.get_obstacle_meta_from_datax(x)
        horizon_steps = int(np.ceil(self.obstacle_horizon_s / self.env.obstacle_traj_dt))

        for i in range(x.shape[0]):
            self.robot.set_joint_position(self.robot.body_joints, x[i, : self.q_dims])
            if not self.robot.check_self_collision_free():
                safe_mask[i] = False
                continue

            traj_i = int(traj_idx[i].item())
            step_i = int(step_idx[i].item())
            self.env.obstacle_traj_idx = traj_i

            is_safe = True
            for k in range(horizon_steps + 1):
                self.env.apply_obstacle_step(step_i + k)
                if self.env.robots_within_distance(self.robot, self.env.obstacle_robot, self.dis_threshold):
                    is_safe = False
                    break
            safe_mask[i] = is_safe

        return safe_mask

    def unsafe_mask(self, x):
        if (
            self.env is None
            or self.env.obstacle_robot is None
            or self.obstacle_qdot_dim == 0
            or self.env.obstacle_traj_dt is None
        ):
            return super().unsafe_mask(x)

        unsafe_mask = torch.zeros_like(x[:, 0]).type_as(x).to(dtype=torch.bool)
        if x.shape[1] <= self.n_dims:
            traj_idx = torch.full((x.shape[0],), float(self.env.obstacle_traj_idx), device=x.device)
            step_idx = torch.full((x.shape[0],), float(self.env.obstacle_traj_step), device=x.device)
        else:
            _, traj_idx, step_idx = self.get_obstacle_meta_from_datax(x)
        horizon_steps = int(np.ceil(self.obstacle_horizon_s / self.env.obstacle_traj_dt))

        for i in range(x.shape[0]):
            self.robot.set_joint_position(self.robot.body_joints, x[i, : self.q_dims])
            traj_i = int(traj_idx[i].item())
            step_i = int(step_idx[i].item())
            self.env.obstacle_traj_idx = traj_i

            is_unsafe = False
            for k in range(horizon_steps + 1):
                self.env.apply_obstacle_step(step_i + k)
                if self.env.robots_in_contact(self.robot, self.env.obstacle_robot):
                    is_unsafe = True
                    break
            unsafe_mask[i] = is_unsafe

        return unsafe_mask


if __name__ == "__main__":
    problem_num = 1000
    obstacle_num = 8

    robot_name = "yumi"
    nominal_params = {"m1": 5.76}
    controller_period = 1 / 30
    simulation_dt = 1 / 120
    environment = ArmEnv([robot_name], GUI=True, config_file="")
    robot = environment.robot_list[0]
    dynamics_model = ArmLidar(
        nominal_params,
        dt=simulation_dt,
        dis_threshold=0.02,
        controller_dt=controller_period,
        env=environment,
        robot=robot,
    )

    while True:
        environment.p.stepSimulation()
