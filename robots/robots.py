import copy
from abc import ABC, abstractmethod
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
    def __init__(self, init_pos=None, speed=1., yaw_rate=.3, dt=1.):
        vel = [speed * np.cos(yaw_rate), speed * np.sin(yaw_rate)]
        super().__init__(init_pos, vel, dt)

        self.speed = speed
        self.yaw_rate = yaw_rate
        self.i = 0

    def update(self):
        self.i += 1
        if self.i < 21:
            self.vel = [self.speed, 0]
            self.pos = self.pos + np.dot(self.vel, self.dt)
            return

        self.vel = [self.speed * np.cos(self.yaw_rate), self.speed * np.sin(self.yaw_rate)]
        self.yaw_rate += .08
        self.pos = self.pos + np.dot(self.vel, self.dt)
