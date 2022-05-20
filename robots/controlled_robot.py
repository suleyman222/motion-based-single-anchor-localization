import matplotlib.pyplot as plt
import numpy as np

from robots.base_robot import BaseRobot2D
from utils import Util


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

    def calc_pos(self, r, v):
        dr = (r - self.prev_r) / self.dt
        self.prev_r = r
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        return pos1, pos2

    def calculate_position(self, prev_pos, r, v):
        dr = (r - self.prev_r) / self.dt
        self.prev_r = r
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        if np.linalg.norm(pos1 - prev_pos) < np.linalg.norm(pos2 - prev_pos):
            return pos1
        else:
            return pos2


def main_known_pos():
    p0 = [3., 1.]
    # try changing direction slightly
    u = [[.001, 0], [.001, 0], [0, .003], [0, .003], [-.002, 0], [-.002, 0], [-.001, -.001], [-.001, -.001]]
    cr = ControlledRobot(u, p0, dt=1)

    pos = [cr.pos]
    measured_pos = [cr.pos]
    for _ in range(len(u)):
        cr.update()
        measured_r, measured_v = cr.get_measurement()
        calc_pos = cr.calculate_position(measured_pos[-1], measured_r, measured_v)
        measured_pos.append(calc_pos)
        pos.append(cr.pos)

    pos = np.array(pos)
    measured_pos = np.array(measured_pos)
    plt.plot(pos[:, 0], pos[:, 1])
    plt.plot(measured_pos[:, 0], measured_pos[:, 1])
    plt.show()


def main():
    p0 = [3., 1.]
    u = [[.001, 0], [.001, 0], [0, .003], [0, .003], [-.002, 0], [-.002, 0], [-.001, -.001], [-.001, -.001]]
    cr = ControlledRobot(u, p0, dt=1)

    pos = [cr.pos]
    estimated_pos1 = []
    estimated_pos2 = []
    for _ in range(len(u)):
        measured_r, measured_v = cr.get_measurement()
        calc_pos1, calc_pos2 = cr.calc_pos(measured_r, measured_v)
        estimated_pos1.append(calc_pos1)
        estimated_pos2.append(calc_pos2)
        cr.update()
        pos.append(cr.pos)

    pos = np.array(pos)
    estimated_pos1 = np.array(estimated_pos1)
    estimated_pos2 = np.array(estimated_pos2)
    plt.plot(pos[:, 0], pos[:, 1])
    plt.plot(estimated_pos1[:, 0], estimated_pos1[:, 1])
    plt.plot(estimated_pos2[:, 0], estimated_pos2[:, 1])
    plt.show()


if __name__ == '__main__':
    # main_known_pos()
    main()
    # x+exp(−0.2x)sin(10x)
    # t = np.linspace(0, 10, )
    # plt.plot(np.sin(3.1*t)+t, t)
    # plt.show()
