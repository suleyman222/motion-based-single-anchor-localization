import numpy as np
from robots.base_robot import BaseRobot2D


class ConstantAccelerationRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, accel=None, dt=1., noise=False, r_std=0., v_std=0.):
        super().__init__(init_pos, init_vel, noise, r_std, v_std, dt)

        if accel is None:
            accel = [.1, .1]

        self.accel = np.array(accel)

    def update(self):
        self.vel = self.vel + self.accel * self.dt
        self.pos = self.pos + self.vel * self.dt
        super().update()


class RandomAccelerationRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, dt=1., noise=False, r_std=0., v_std=0., ax_noise=.2, ay_noise=.1):
        super().__init__(init_pos, init_vel, noise, r_std, v_std, dt)
        self.ay_noise = ay_noise
        self.ax_noise = ax_noise
        self.accel = None

    def update(self):
        self.accel = np.dot([np.random.randn() * self.ax_noise, np.random.randn() * self.ay_noise], self.dt)
        self.vel = self.vel + np.dot(self.accel, self.dt)
        self.pos = self.pos + np.dot(self.vel, self.dt)
        super().update()
