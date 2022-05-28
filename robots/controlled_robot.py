import copy

import numpy as np
from robots.base_robot import BaseRobot2D


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
