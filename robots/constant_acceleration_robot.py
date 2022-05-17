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
        self.all_positions.append(self.pos)
