import filterpy.common
import numpy as np

from utils import Util
from robots.BaseRobot2D import BaseRobot2D
from robots.ConstantAccelerationRobot2D import ConstantAccelerationRobot2D

# Ideas to check out:
# - a changing value of dt
# - different data rates of sensors


class PositionTracking:
    def __init__(self, kf, robot: BaseRobot2D = ConstantAccelerationRobot2D(), count=50):
        self.robot = robot
        self.count = count

        self.dt = robot.dt

        self.prev_r = np.linalg.norm(robot.pos)
        self.kf = kf

    def run(self):
        init_pos = self.robot.pos
        estimated_positions = [init_pos]
        measured_positions = [init_pos]

        for _ in range(self.count):
            self.robot.update()
            measured_r, measured_v = self.robot.get_measurement()
            measured_pos = self.calculate_position(estimated_positions[-1], measured_r, measured_v)
            measured_positions.append(measured_pos)

            self.kf.predict()
            self.kf.update(measured_pos)
            estimated_pos = [self.kf.x[0], self.kf.x[1]]
            estimated_positions.append(estimated_pos)

            # TODO: How do we update prev_r?
            # self.prev_r = np.linalg.norm(estimated_pos)
            self.prev_r = measured_r

        Util.plot_path(np.array(self.robot.all_positions), np.array(measured_positions), np.array(estimated_positions))

    def calculate_position(self, prev_pos, r, v):
        dr = (r - self.prev_r) / self.dt
        s = np.linalg.norm(v)
        alpha = np.arctan(v[1] / v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        if np.linalg.norm(pos1 - prev_pos) < np.linalg.norm(pos2 - prev_pos):
            return pos1
        else:
            return pos2
