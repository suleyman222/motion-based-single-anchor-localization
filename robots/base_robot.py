from abc import ABC, abstractmethod
import robots.accelerating_robots as rob
import numpy as np


class BaseRobot2D(ABC):
    def __init__(self, init_pos=None, init_vel=None, noise=False, r_std=0., v_std=0., dt=1.):
        if init_pos is None:
            init_pos = [0., 0.]
        if init_vel is None:
            init_vel = [0., 0.]

        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.all_positions = [self.pos]

        self.noise = noise
        self.r_std = r_std
        self.v_std = v_std
        self.dt = dt

    def get_measurement(self):
        # TODO: self.vel or self.vel * self.dt
        v = self.vel
        r = np.linalg.norm(self.pos)

        if self.noise:
            r += np.random.normal(0, self.r_std)
            v = v + np.random.normal(0, self.v_std)
            # v = [v[0] + self.v_std * np.random.randn(), v[1] + 1.5 * self.v_std * np.random.randn()]
        return r, v

    @abstractmethod
    def update(self):
        # TODO: Move saving position to TwoRobotSystem
        self.all_positions.append(self.pos)


class TwoRobotSystem:
    def __init__(self, anchor_robot: BaseRobot2D, target_robot: BaseRobot2D, noise=False, r_std=0., v_std=0.):
        if anchor_robot is None:
            anchor_robot = rob.ConstantAccelerationRobot2D([0., 0.], [0., 0.], [0., 0.], dt=target_robot.dt)

        self.v_std = v_std
        self.r_std = r_std
        self.noise = noise
        self.anchor_robot = anchor_robot
        self.target_robot = target_robot
        self.dt = target_robot.dt

        if anchor_robot.dt != target_robot.dt:
            print("Target and anchor dt are different")

    def update(self):
        self.target_robot.update()
        self.anchor_robot.update()

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



