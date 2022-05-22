import matplotlib.pyplot as plt
import numpy as np

from robots.base_robot import BaseRobot2D
from utils import Util


def cos_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def closest_to(target, options):
    distance = float("inf")
    closest = None
    for option in options:
        d = np.linalg.norm(option - target)
        if d < distance:
            closest = option
            distance = d
    return closest


class ControlledRobot(BaseRobot2D):
    def __init__(self, control_input, init_pos=None, init_vel=None, dt=1., noise=False, r_std=0., v_std=0.):
        super().__init__(init_pos, init_vel, noise, r_std, v_std, dt)
        self.localized = False
        self.control_input = control_input
        self.prev_r = np.linalg.norm(self.pos)

    def update(self):
        if self.control_input:
            self.vel = self.control_input.pop(0)
            self.pos = self.pos + np.dot(self.vel, self.dt)
            super().update()

    def calc_pos(self, r, v):
        dr = (r - self.prev_r) / self.dt
        self.prev_r = r
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        return [pos1, pos2]


def main_known_pos():
    p0 = [3., 2.]
    # u = [[1, 0], [1, 0], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2],
    #      [1, 2], [1, 2]]
    u = [[1, 0]] * 100 + [[1, 2]] * 100
    cr = ControlledRobot(u, p0, dt=.1)

    pos = [cr.pos]
    measured_pos = [cr.pos]

    for _ in range(len(u)):
        cr.update()
        measured_r, measured_v = cr.get_measurement()
        calc_pos = closest_to(measured_pos[-1], cr.calc_pos(measured_r, measured_v))
        measured_pos.append(calc_pos)
        pos.append(cr.pos)

    pos = np.array(pos)
    measured_pos = np.array(measured_pos)

    plt.plot(pos[:, 0], pos[:, 1])
    plt.plot(measured_pos[:, 0], measured_pos[:, 1])
    plt.show()

    print(Util.rmse(measured_pos, pos))


def main():
    p0 = [3., 2.]
    u = [[1, 0]] * 100 + [[1, 2]] * 100
    cr = ControlledRobot(u, p0, dt=.1)

    alt1 = []
    alt2 = []
    measured_positions = []
    prev_v = None

    for _ in range(len(u)):
        cr.update()
        measured_r, measured_v = cr.get_measurement()
        if cr.localized:
            measured_pos = closest_to(measured_positions[-1], cr.calc_pos(measured_r, measured_v))
            measured_positions.append(measured_pos)
        elif prev_v:
            [pos1, pos2] = cr.calc_pos(measured_r, measured_v)
            alt1.append(pos1)
            alt2.append(pos2)

            similarity = cos_similarity(prev_v, measured_v)
            if similarity < .5:
                cr.localized = True
                prev1 = alt1[-1]
                prev2 = alt2[-1]

                max_sim = -2
                max_pos = None
                for pos in [pos1, pos2]:
                    for prev in [prev1, prev2]:
                        sim = cos_similarity(pos, prev)
                        if sim > max_sim:
                            max_sim = sim
                            max_pos = pos
                measured_positions.append(max_pos)
        prev_v = measured_v

    pos = np.array(cr.all_positions)
    alt1 = np.array(alt1)
    alt2 = np.array(alt2)
    measured_positions = np.array(measured_positions)

    plt.plot(pos[:, 0], pos[:, 1])
    plt.plot(alt1[:, 0], alt1[:, 1])
    plt.plot(alt2[:, 0], alt2[:, 1])
    plt.plot(measured_positions[:, 0], measured_positions[:, 1])
    plt.show()


if __name__ == '__main__':
    # main_known_pos()
    main()
