from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt

from robots.accelerating_robots import ConstantAccelerationRobot2D, RandomAccelerationRobot2D
from robots.base_robot import BaseRobot2D
from robots.controlled_robot import ControlledRobot2D
from utils import Util


class BaseLocalization(ABC):
    def __init__(self, robot: BaseRobot2D, count=50):
        self.robot = robot
        self.count = count
        self.dt = robot.dt
        self.prev_r = np.linalg.norm(robot.pos)

    def calculate_possible_positions(self, r, v):
        dr = (r - self.prev_r) / self.dt
        self.prev_r = r
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = [r * np.cos(alpha + theta), r * np.sin(alpha + theta)]
        pos2 = [r * np.cos(alpha - theta), r * np.sin(alpha - theta)]
        return [pos1, pos2]

    @abstractmethod
    def run(self):
        pass


class PositionTracking(BaseLocalization):
    def __init__(self, kf, robot: BaseRobot2D = ConstantAccelerationRobot2D(), count=50):
        super().__init__(robot, count)
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

            # self.kf.predict()
            # self.kf.update(measured_pos)
            # estimated_pos = [self.kf.x[0], self.kf.x[1]]
            # estimated_positions.append(estimated_pos)

            # TODO: How do we update prev_r?
            # self.prev_r = np.linalg.norm(estimated_pos)
            self.prev_r = measured_r
        return np.array(measured_positions), np.array(estimated_positions)

    def calculate_position(self, prev_pos, r, v):
        return Util.closest_to(prev_pos, self.calculate_possible_positions(r, v))


class MotionBasedLocalization(BaseLocalization):
    def __init__(self, robot: BaseRobot2D, count=50):
        super().__init__(robot, count)
        self.localized = False
        self.idx_localized = None

        # Every measurement we get two possible locations for the robot. The initial position is not available
        # through measurements, since the algorithm makes use of change in distance.
        self.measured_positions = [([None, None], [None, None])] * (self.count + 1)
        self.chosen_positions = [[None, None]] * (self.count + 1)

        # TODO: noise and filtering

    def run(self):
        def find_max_similarity(prev_positions, new_positions):
            max_sim = -2
            max_position = None
            for pos in new_positions:
                for prev in prev_positions:
                    sim = Util.cos_similarity(pos, prev)
                    if sim > max_sim:
                        max_sim = sim
                        max_position = pos
            return max_position

        prev_v = [None, None]
        for i in range(self.count):
            self.robot.update()
            measured_r, measured_v = self.robot.get_measurement()
            [pos1, pos2] = self.calculate_possible_positions(measured_r, measured_v)
            self.measured_positions[i + 1] = (pos1, pos2)

            if i == 0:
                prev_v = measured_v
                continue

            if self.localized:
                measured_pos = Util.closest_to(self.chosen_positions[i], [pos1, pos2])
                self.chosen_positions[i+1] = measured_pos
            else:
                similarity = Util.cos_similarity(prev_v, measured_v)
                if similarity < .99:
                    self.localized = True
                    prev1 = self.measured_positions[i][0]
                    prev2 = self.measured_positions[i][1]

                    max_pos = find_max_similarity([prev1, prev2], [pos1, pos2])
                    self.chosen_positions[i + 1] = max_pos
                    self.idx_localized = i + 1
            prev_v = measured_v


if __name__ == '__main__':
    p0 = [3., 2.]
    u = [[1, 0]] * 100 + [[1, 2]] * 100
    # u = [[1, 0], [1, 0], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    #      [1, 2], [1, 2]]
    cr = ControlledRobot2D(u, p0, dt=.1, noise=False, r_std=0.001, v_std=0.001)
    # cr = ConstantAccelerationRobot2D(p0, [.1, .1], [.1, .1], dt=.1)
    # cr = RandomAccelerationRobot2D(p0, [1, 1], .1, ax_noise=10, ay_noise=1)

    loc = MotionBasedLocalization(cr, len(u))
    loc.run()

    if not loc.localized:
        plt.title("Couldn't localize")

    for i in reversed(range(loc.idx_localized)):
        if i == 0:
            continue
        loc.chosen_positions[i] = Util.closest_to(loc.chosen_positions[i+1], loc.measured_positions[i])

    all_pos = np.array(cr.all_positions)
    alt1 = np.array([possible_position[0] for possible_position in loc.measured_positions])
    alt2 = np.array([possible_position[1] for possible_position in loc.measured_positions])
    chosen_positions = np.array(loc.chosen_positions)

    plt.plot(all_pos[:, 0], all_pos[:, 1])
    plt.plot(alt1[:, 0], alt1[:, 1])
    plt.plot(alt2[:, 0], alt2[:, 1])
    plt.plot(chosen_positions[:, 0], chosen_positions[:, 1])

    # pt = PositionTracking(None, cr, len(u))
    # m, e = pt.run()
    # measuredp = np.array(m)
    # plt.plot(measuredp[:, 0], measuredp[:, 1])

    plt.show()
