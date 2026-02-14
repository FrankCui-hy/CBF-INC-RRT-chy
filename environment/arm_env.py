import os
import numpy as np
import pybullet_utils.bullet_client as bc
import pybullet
import warnings
import pybullet_data

from environment.franka_panda import FrankaPanda
from environment.magician import Magician


def sample_surface(center, half_extent, idx, num, add_normal=False):
	a = np.zeros((num, 3))
	b = np.random.uniform(low=(-1, -1, -1), high=(1, 1, 1), size=(num, 3))
	for ii in range(3):
		a[:, ii] = center[ii] + half_extent[ii] * b[:, ii]
	a[:, idx // 2] = center[idx // 2] + half_extent[idx // 2] * np.sign(idx % 2 - 0.5)
	if add_normal:
		a = np.concatenate((a, np.sign(idx % 2 - 0.5) * np.eye(3)[idx // 2].reshape(1, 3).repeat(num, axis=0)), axis=1)
	return a


class ArmEnv:
	'''
	Separate arm interface with maze environment ones
	'''

	def __init__(
			self,
			robot_name_list,
			config_file: str,
			GUI=False,
			include_floor=True,
			obstacle_robot_name: str = None,
			obstacle_traj_path: str = None,
			obstacle_robot_base_pos: tuple = (0.3, 0.0, 0.0),
			obstacle_robot_base_orn: tuple = (0.0, 0.0, 0.0, 1.0),
	):
		print("Initializing environment...")

		self.robot_name_list = robot_name_list
		self.robot_list = []
		self.obstacle_robot = None
		self.obstacle_robot_name = obstacle_robot_name
		self.obstacle_robot_base_pos = obstacle_robot_base_pos
		self.obstacle_robot_base_orn = obstacle_robot_base_orn
		self.obstacle_traj = None
		self.obstacle_traj_dt = None
		self.obstacle_traj_idx = 0
		self.obstacle_traj_step = 0
		self.obstacle_traj_time_accum = 0.0
		self.obstacle_qdot = None
		self.obstacle_ids = []

		self.include_floor = include_floor
		print(f"included floor: {self.include_floor}")

		if GUI:
			self.p = bc.BulletClient(connection_mode=pybullet.GUI,
									 options='--background_color_red=1.0 --background_color_green=1.0 --background_color_blue=1.0')
		else:
			self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)

		self.use_training_env = (len(config_file) == 0)
		if self.use_training_env:
			# training environment file
			self.env_config_file = f'{str.rsplit(os.path.abspath(__file__), "/", 2)[0]}/models/env_file/env_600_4.npz'
			self.obstacle_num = 4
		else:
			self.env_config_file = config_file

		if not os.path.exists(self.env_config_file) and self.use_training_env:
			self._generate_env_config(self.env_config_file, 4)

		self.env_config = np.load(self.env_config_file, allow_pickle=True)

		self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
		self.reset_env(self.get_env_config(-1), enable_object=False, obstacle_robot_name=obstacle_robot_name)

		if obstacle_traj_path:
			self.load_obstacle_trajectories(obstacle_traj_path)

	def __str__(self):
		return self.robot_name_list

	def reset_env(self, obs_configs, enable_object=False, obstacle_robot_name: str = None):
		self.p.resetSimulation()
		self.robot_list = []
		assert len(self.robot_name_list) <= 1, "Only support one robot now."

		if self.include_floor:
			plane = self.p.createCollisionShape(self.p.GEOM_PLANE)
			self.plane = self.p.createMultiBody(0, plane)

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			for robot_name in self.robot_name_list:
				if robot_name == "panda":
					self.robot_list.append(FrankaPanda(self.p))
				elif robot_name == "magician":
					self.robot_list.append(Magician(self.p))
				else:
					raise NotImplementedError(f"Robot {robot_name} not supported yet.")

				if self.include_floor:
					self.p.setCollisionFilterPair(self.robot_list[-1].robotId, self.plane, -1, -1, 0)
					self.p.setCollisionFilterPair(self.robot_list[-1].robotId, self.plane, 1, -1, 0)

			if obstacle_robot_name is None:
				obstacle_robot_name = self.obstacle_robot_name

			if obstacle_robot_name is not None:
				if obstacle_robot_name == "panda":
					self.obstacle_robot = FrankaPanda(self.p)
				elif obstacle_robot_name == "magician":
					self.obstacle_robot = Magician(self.p)
				else:
					raise NotImplementedError(f"Obstacle robot {obstacle_robot_name} not supported.")

				self.p.resetBasePositionAndOrientation(
					self.obstacle_robot.robotId,
					self.obstacle_robot_base_pos,
					self.obstacle_robot_base_orn,
				)

				if self.include_floor:
					self.p.setCollisionFilterPair(self.obstacle_robot.robotId, self.plane, -1, -1, 0)
					self.p.setCollisionFilterPair(self.obstacle_robot.robotId, self.plane, 1, -1, 0)

		if enable_object:
			if len(self.robot_list) > 1:
				raise NotImplementedError
			else:
				object = self.p.createCollisionShape(self.p.GEOM_CYLINDER, radius=0.01,
													 height=np.random.uniform(0.15, 0.25))
				objectPos, objectOrn = self.robot_list[0].get_link_PosOrn(self.robot_list[0].body_joints[-1])
				objectOrn = \
					self.p.multiplyTransforms([0, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0, 0], [0, 0, 0], objectOrn)[
						1]
				self.object = self.p.createMultiBody(0.01, object, basePosition=objectPos, baseOrientation=objectOrn)
				self.robot_list[0].end_effector.activate(self.object)

		self._generate_obstacle(obs_configs=obs_configs)

	# p.setGravity(0, 0, -10)

	def get_env_config(self, idx):
		# return obstacle positions and sizes
		if idx >= 0:
			obs_positions = self.env_config['obstacle_positions'][idx]
			if 'obstacle_sizes' in self.env_config.keys():
				obs_sizes = self.env_config['obstacle_sizes'][idx]
			else:
				obs_sizes = np.array([[0.05, 0.05, 0.1] for _ in range(obs_positions.shape[0])])
			return obs_positions, obs_sizes
		else:
			return np.array([[0.3, 0.17, 0.3], [-0.4, 0.23, 0.4], [0.0, -0.23, 0.6]]), \
				np.array([[0.05, 0.05, 0.1], [0.05, 0.05, 0.1], [0.05, 0.05, 0.1]])

	def sample_obstacle_surface(self, total_num, add_normal=False):
		# note: no self-collision information
		if self.obstacle_robot is not None:
			return self.sample_robot_surface(self.obstacle_robot, total_num, add_normal=add_normal)
		for_floor = 0
		each_obstacle = np.random.randint(low=0, high=len(self.obstacle_ids) - int(self.include_floor),
										  size=total_num - for_floor)
		# points on floor
		points_global = np.random.uniform(low=(-1, -1), high=(1, 1), size=(for_floor, 2))
		points_global = np.concatenate((points_global, np.zeros((for_floor, 1 + 3 * int(add_normal)))), axis=1)

		# points on obstacles
		for obs_idx in range(len(self.obstacle_ids) - int(self.include_floor)):
			obs_position = self.obs_positions[obs_idx]
			obs_size = self.obs_sizes[obs_idx]
			rand_idx = np.random.randint(0, 6, np.where(each_obstacle == obs_idx)[0].shape[0])
			for jj in range(6):
				new_array = sample_surface(obs_position, obs_size, jj, np.where(rand_idx == jj)[0].shape[0],
										   add_normal=add_normal)
				points_global = np.concatenate((points_global, new_array), axis=0)
		return points_global

	def sample_robot_surface(self, robot, total_num, add_normal=False):
		"""
		Approximate robot surface by sampling points on each link's AABB.
		"""
		link_ids = robot.body_joints
		points_global = np.zeros((0, 3 + 3 * int(add_normal)), dtype=np.float32)
		if len(link_ids) == 0:
			return points_global

		# Uniformly choose links to sample
		link_choices = np.random.randint(low=0, high=len(link_ids), size=total_num)
		for link_idx, link_id in enumerate(link_ids):
			which = np.where(link_choices == link_idx)[0]
			if which.shape[0] == 0:
				continue
			aabb_min, aabb_max = self.p.getAABB(robot.robotId, link_id)
			aabb_min = np.array(aabb_min)
			aabb_max = np.array(aabb_max)
			center = (aabb_min + aabb_max) / 2.0
			half = (aabb_max - aabb_min) / 2.0
			rand_faces = np.random.randint(0, 6, which.shape[0])
			for face_idx in range(6):
				num = np.sum(rand_faces == face_idx)
				if num == 0:
					continue
				samples = sample_surface(center, half, face_idx, num, add_normal=add_normal)
				points_global = np.concatenate((points_global, samples), axis=0)

		return points_global

	def load_obstacle_trajectories(self, traj_path: str):
		data = np.load(traj_path, allow_pickle=True)
		self.obstacle_traj = data
		self.obstacle_traj_dt = float(data["dt"])
		self.obstacle_traj_idx = 0
		self.obstacle_traj_step = 0
		self.obstacle_traj_time_accum = 0.0
		self.obstacle_qdot = np.zeros_like(data["q_trajs"][0, 0, :], dtype=np.float32)

	def set_obstacle_trajectory(self, traj_idx: int, step_idx: int = 0):
		self.obstacle_traj_idx = traj_idx
		self.obstacle_traj_step = step_idx
		self.obstacle_traj_time_accum = 0.0
		self.apply_obstacle_step(step_idx)

	def apply_obstacle_step(self, step_idx: int):
		if self.obstacle_robot is None or self.obstacle_traj is None:
			return
		q_trajs = self.obstacle_traj["q_trajs"]
		qdot_trajs = self.obstacle_traj["qdot_trajs"]
		step_idx = int(step_idx % q_trajs.shape[1])
		q = q_trajs[self.obstacle_traj_idx, step_idx]
		qdot = qdot_trajs[self.obstacle_traj_idx, step_idx]
		self.obstacle_robot.set_joint_position(self.obstacle_robot.body_joints, q)
		self.obstacle_qdot = qdot
		self.obstacle_traj_step = step_idx

	def step_obstacle(self, num_steps: int = 1, sim_dt: float = None):
		"""Advance obstacle trajectory using real time.

		If sim_dt is provided, accumulate time and advance trajectory only when
		(trajectory_dt) is reached. This prevents speeding up obstacle motion when
		simulation dt is smaller than trajectory dt.
		"""
		if self.obstacle_robot is None or self.obstacle_traj is None:
			return
		if sim_dt is None or self.obstacle_traj_dt is None:
			next_step = self.obstacle_traj_step + num_steps
			self.apply_obstacle_step(next_step)
			return
		self.obstacle_traj_time_accum += float(sim_dt) * int(num_steps)
		if self.obstacle_traj_time_accum < self.obstacle_traj_dt:
			return
		steps_to_advance = int(self.obstacle_traj_time_accum / self.obstacle_traj_dt)
		self.obstacle_traj_time_accum -= steps_to_advance * self.obstacle_traj_dt
		next_step = self.obstacle_traj_step + steps_to_advance
		self.apply_obstacle_step(next_step)

	def get_obstacle_qdot(self):
		if self.obstacle_qdot is None:
			if self.obstacle_robot is None:
				return None
			return np.zeros((self.obstacle_robot.body_dim,), dtype=np.float32)
		return self.obstacle_qdot

	def robots_within_distance(self, robot_a, robot_b, distance: float) -> bool:
		# robot_a, robot_b are BasicRobot instances
		self.p.performCollisionDetection()
		for link_idx in range(-1, robot_a.n_joints):
			points = self.p.getClosestPoints(robot_a.robotId, robot_b.robotId, distance, linkIndexA=link_idx)
			if bool(points):
				return True
		return False

	def robots_in_contact(self, robot_a, robot_b) -> bool:
		self.p.performCollisionDetection()
		points = self.p.getContactPoints(robot_a.robotId, robot_b.robotId)
		return len(points) > 0

	# ===================== internal module ===========================

	def _generate_env_config(self, file_name, obstacle_num, problem_num=1000):
		# generate or load test cases
		if len(self.robot_name_list) > 1:
			raise NotImplementedError('Environment config with multiple arms not implemented.')
		else:
			positions = np.zeros([0, obstacle_num, 3])

			while positions.shape[0] < problem_num:
				try:
					position = np.random.uniform(low=(-0.5, -0.5, 0), high=(0.5, 0.5, 1), size=(obstacle_num, 3))
					positions = np.concatenate([positions, np.expand_dims(position, axis=0)], axis=0)
				except AssertionError:
					continue
			assert file_name.endswith('.npz')
			np.savez(file_name, obstacle_positions=positions)

	def _generate_obstacle(self, obs_configs):
		if self.obstacle_robot is not None:
			# Dynamic obstacle: another robot
			self.obs_positions, self.obs_sizes = obs_configs
			self.obstacle_ids = []
			if self.include_floor:
				self.obstacle_ids.append(self.plane)
			self.obstacle_ids.append(self.obstacle_robot.robotId)
			return

		self.obs_positions, self.obs_sizes = obs_configs
		if self.include_floor:
			self.obstacle_ids = [self.plane]
		else:
			self.obstacle_ids = []
		for obs_position, obs_size in zip(self.obs_positions, self.obs_sizes):
			halfExtents = obs_size
			basePosition = obs_position
			baseOrientation = [0, 0, 0, 1]
			self.obstacle_ids.append(self._create_voxel(halfExtents, basePosition, baseOrientation, color='random'))

	def _create_voxel(self, halfExtents, basePosition, baseOrientation, color='random'):
		voxColId = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=halfExtents)
		if color == 'random':
			voxVisID = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
												rgbaColor=[58 / 256, 107 / 256, 53 / 256, 1],
												# np.random.uniform(0, 1, size=3).tolist() + [0.8],
												halfExtents=halfExtents)
		else:
			voxVisID = self.p.createVisualShape(shapeType=self.p.GEOM_BOX,
												rgbaColor=color,
												halfExtents=halfExtents)
		voxId = self.p.createMultiBody(baseMass=0,
									   baseCollisionShapeIndex=voxColId,
									   baseVisualShapeIndex=voxVisID,
									   basePosition=basePosition,
									   baseOrientation=baseOrientation)
		return voxId


if __name__ == '__main__':
	nominal_params = {"m1": 5.76}
	controller_period = 1 / 30
	simulation_dt = 1 / 120
	environment = ArmEnv(['panda'], GUI=True, config_file='')

	while True:
		environment.p.stepSimulation()
		print(environment.robot_list[0].check_self_collision_free())
