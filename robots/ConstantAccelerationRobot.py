import numpy as np
from matplotlib import pyplot as plt
from robots.BaseRobot import Robot
from utils import Util


class ConstantAccelerationRobot(Robot):
    def __init__(self, init_pos=None, init_vel=None, accel=None, dt=1., noise=False, r_std=0., v_std=0.):
        super().__init__()

        if accel is None:
            accel = [.1, .1]
        if init_vel is None:
            init_vel = [1., 1.]
        if init_pos is None:
            init_pos = [0., 0.]

        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.accel = np.array(accel)
        self.distance = np.linalg.norm(self.pos)
        self.all_positions = [self.pos]
        self.dt = dt

        self.noise = noise
        self.r_std = r_std
        self.v_std = v_std

    def update(self):
        self.vel = self.vel + self.accel * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.all_positions.append(self.pos)

    def get_position_measurement(self):
        v = self.vel

        # Calculate and update distance
        r = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
        dr = (r - self.distance) / self.dt
        self.distance = r

        if self.noise:
            r += self.r_std * np.random.randn() * self.dt
            v = v + self.v_std * np.random.randn() * self.dt

        s = np.linalg.norm(v)
        alpha = np.arctan(v[1] / v[0])
        theta = np.arccos(Util.clamp(dr / s, -1, 1))

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        if np.linalg.norm(pos1-self.pos) < np.linalg.norm(pos2-self.pos):
            self.last_measured_pos = pos1
            return pos1
        else:
            self.last_measured_pos = pos2
            return pos2


if __name__ == '__main__':
    pos0 = [1., 2.]
    v0 = [1., 3.]
    acc = [.5, .7]

    rob = ConstantAccelerationRobot(init_pos=pos0, init_vel=v0, accel=acc, dt=0.1, r_std=.1, v_std=.1, noise=True)
    initial_est_pos = rob.get_position_measurement()
    xs_est, ys_est = [pos0[0]], [pos0[1]]

    count = 100
    for _ in range(count):
        rob.update()
        est_pos = rob.get_position_measurement()
        xs_est.append(est_pos[0])
        ys_est.append(est_pos[1])

    # Plot results
    real_positions = np.array(rob.all_positions)
    plt.plot(xs_est, ys_est, label='Measured position')
    plt.plot(real_positions[:, 0], real_positions[:, 1], label='Real position')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.show()
