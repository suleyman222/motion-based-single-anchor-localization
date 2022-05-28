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
        super().update()


class RandomAccelerationRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, dt=1., ax_noise=.2, ay_noise=.1):
        super().__init__(init_pos, init_vel, dt)
        self.ay_noise = ay_noise
        self.ax_noise = ax_noise
        self.accel = None

    def update(self):
        self.accel = np.dot([np.random.randn() * self.ax_noise, np.random.randn() * self.ay_noise], self.dt)
        self.vel = self.vel + np.dot(self.accel, self.dt)
        self.pos = self.pos + np.dot(self.vel, self.dt)
        super().update()


class ControlledRobot2D(BaseRobot2D):
    def __init__(self, control_input,  dt=1., init_pos=None, init_vel=None):
        super().__init__(init_pos, init_vel, dt)
        self.localized = False
        self.control_input = copy.deepcopy(control_input)
        self.prev_r = np.linalg.norm(self.pos)

    def update(self):
        if self.control_input:
            self.vel = np.array(self.control_input.pop(0))
            self.pos = self.pos + np.dot(self.vel, self.dt)
            super().update()


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

        if anchor_robot.dt != target_robot.dt:
            print("Target and anchor dt are different")

    def update(self):
        self.target_robot.update()
        self.anchor_robot.update()
        self.all_anchor_positions.append(self.anchor_robot.pos)
        self.all_target_positions.append(self.target_robot.pos)

    def get_measurement(self):
        # TODO: Change to return v_anchor, v_target and r
        v_tracked_robot = self.target_robot.vel
        v_anchor_robot = self.anchor_robot.vel if self.anchor_robot else np.zeros(2)

        v = v_tracked_robot - v_anchor_robot
        r = np.linalg.norm(self.target_robot.pos - self.anchor_robot.pos)

        if self.noise:
            r += np.random.normal(0, self.r_std)
            v = v + np.random.normal(0, self.v_std)
            # v = [v[0] + self.v_std * np.random.randn(), v[1] + 1.5 * self.v_std * np.random.randn()]
        return r, v
