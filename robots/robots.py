import copy
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseRobot2D(ABC):
    def __init__(self, init_pos=None, init_vel=None, dt=1.):
        if init_pos is None:
            init_pos = [0., 0.]
        if init_vel is None:
            init_vel = [0., 0.]

        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.dt = dt

    @abstractmethod
    def update(self):
        pass


class ConstantAccelerationRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, accel=None, dt=1.):
        super().__init__(init_pos, init_vel, dt)

        if accel is None:
            accel = [.1, .1]

        self.accel = np.array(accel)

    def update(self):
        self.vel = self.vel + self.accel * self.dt
        self.pos = self.pos + self.vel * self.dt


class RandomAccelerationRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, dt=1., ax_noise=.2, ay_noise=.1):
        super().__init__(init_pos, init_vel, dt)
        self.ay_noise = ay_noise
        self.ax_noise = ax_noise
        self.accel = None

    def update(self):
        # self.accel = np.dot([np.random.randn() * self.ax_noise, np.random.randn() * self.ay_noise], self.dt)
        self.accel = [np.random.randn() * self.ax_noise, np.random.randn() * self.ay_noise]
        self.vel = self.vel + np.dot(self.accel, self.dt)
        self.pos = self.pos + np.dot(self.vel, self.dt)


class ControlledRobot2D(BaseRobot2D):
    def __init__(self, control_input,  dt=1., init_pos=None, init_vel=None):
        super().__init__(init_pos, init_vel, dt)
        self.control_input = copy.deepcopy(control_input)

    def update(self):
        if self.control_input:
            self.vel = np.array(self.control_input.pop(0))
            self.pos = self.pos + np.dot(self.vel, self.dt)


class RotatingRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, dt=1.):
        # TODO: these func params dont do anything
        super().__init__(init_pos, init_vel, dt)

        self.speed = .1  # [m/s]
        self.yaw_rate = .15
        self.vel = [self.speed * np.cos(self.yaw_rate), self.speed * np.sin(self.yaw_rate)]
        self.i = 0

    def update(self):
        self.i += 1
        if self.i < 50:
            self.vel = [1 * self.speed, 0]
            self.pos = self.pos + np.dot(self.vel, self.dt)
            return

        self.vel = [self.speed * np.cos(self.yaw_rate), self.speed * np.sin(self.yaw_rate)]
        self.yaw_rate += .01
        self.pos = self.pos + np.dot(self.vel, self.dt)


class TwoRobotSystem:
    def __init__(self, anchor_robot: Optional[BaseRobot2D], target_robot: BaseRobot2D, noise=False, r_std=0., v_std=0.):
        if anchor_robot is None:
            anchor_robot = ConstantAccelerationRobot2D([0., 0.], [0., 0.], [0., 0.], dt=target_robot.dt)

        self.v_std = v_std
        self.r_std = r_std
        self.noise = noise
        self.anchor_robot = anchor_robot
        self.target_robot = target_robot
        self.dt = target_robot.dt

        self.all_anchor_positions = [anchor_robot.pos]
        self.all_target_positions = [target_robot.pos]

        # Measurements
        self.real_r = []
        self.measured_r = []
        self.measured_v = []

        if anchor_robot.dt != target_robot.dt:
            print("Target and anchor dt are different!")

    def update(self):
        self.target_robot.update()
        self.anchor_robot.update()
        self.all_anchor_positions.append(self.anchor_robot.pos)
        self.all_target_positions.append(self.target_robot.pos)

    def get_v_measurement(self):
        # TODO: Change to return v_anchor, v_target
        v_tracked_robot = self.target_robot.vel
        v_anchor_robot = self.anchor_robot.vel
        v = v_tracked_robot - v_anchor_robot

        if self.noise:
            v = v + np.random.normal(0, self.v_std)
            # v = [v[0] + self.v_std * np.random.randn(), v[1] + 1.5 * self.v_std * np.random.randn()]
        self.measured_v.append(v)
        return v

    def get_r_measurement(self):
        r = np.linalg.norm(self.target_robot.pos - self.anchor_robot.pos)
        self.real_r.append(r)

        if self.noise:
            path_loss_exp = 2
            p_0 = -52
            p_ij = p_0 - 10 * path_loss_exp * np.log10(r)
            sigma_rssi = 6  # [dB]

            M = 100
            noisy_rssi = 0
            for i in range(M):
                noisy_rssi += p_ij + np.random.normal(0, sigma_rssi)
            noisy_rssi /= M

            noisy_distance = 10**((-52 - noisy_rssi) / (10 * path_loss_exp))
            # print(r, noisy_distance)

            # eta = 2
            # M = 20
            # sigma_k = 3
            # std = np.sqrt(r**2 * sigma_k**2 * (np.log(10) / (10 * eta))**2 / M)
            # print(r, std)
            r += np.random.normal(0, self.r_std)

            # r = noisy_distance
        self.measured_r.append(r)
        return r
