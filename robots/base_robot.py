from abc import ABC, abstractmethod

import numpy as np


class BaseRobot2D(ABC):
    def __init__(self, init_pos=None, init_vel=None, noise=False, r_std=0., v_std=0., dt=1.):
        if init_pos is None:
            init_pos = [0., 0.]
        if init_vel is None:
            init_vel = [1., 1.]

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
        self.all_positions.append(self.pos)
    