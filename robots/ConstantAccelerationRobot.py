import numpy as np
from matplotlib import pyplot as plt
from robots.BaseRobot import Robot
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


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
        r = np.sqrt(self.pos[0]**2 + self.pos[1]**2)
        v = self.vel
        if self.noise:
            r += self.r_std * np.random.randn() * self.dt
            v = v + self.v_std * np.random.randn() * self.dt

        dr = (r - self.distance) / self.dt
        s = np.linalg.norm(v)
        self.distance = r

        alpha = np.arctan(v[1] / v[0])
        theta = np.arccos(dr / s)

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        if np.linalg.norm(pos1-self.pos) < np.linalg.norm(pos2-self.pos):
            return pos1
        else:
            return pos2


if __name__ == '__main__':
    pos0 = [1., 2.]
    v0 = [1., 3.]
    acc = [.5, .6]

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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) # type:figure.Figure, ((axes.Axes, axes.Axes,), (axes.Axes, axes.Axes,))
    plt.subplots_adjust(hspace=0.7)
    plt.suptitle('Localization without noise, known initial position, constant acceleration')

    # Plot X coordinates
    ax1.plot(real_positions[:, 0], label='real x')
    ax1.plot(xs_est, label='estimated x')
    ax1.set_title('X position')
    ax1.set(ylabel='x coordinate')
    # print(np.sum(np.array(rob.xs) - np.array(xs_est)))

    ax2.plot(real_positions[:, 0], label='real x')
    ax2.plot(xs_est, label='estimated x')
    ax2.set_title('X position (zoomed in)')
    ax2.set_xlim(20, 21)
    ax2.set_ylim(4, 4.3)
    ax2.legend()

    # Plot Y coordinates
    ax3.plot(real_positions[:, 1], label='real y', color='red')
    ax3.plot(ys_est, label='estimated y', color='green')
    ax3.set_title('Y position')
    ax3.set(xlabel='time', ylabel='y coordinate')
    # print(np.sum(np.array(rob.ys) - np.array(ys_est)))

    ax4.plot(real_positions[:, 1], label='real y', color='red')
    ax4.plot(ys_est, label='estimated y', color='green')
    ax4.set_title('Y position (zoomed in)')
    ax4.set(xlabel='time')
    ax4.legend()
    ax4.set_xlim(16, 16.5)
    ax4.set_ylim(7.5, 8)

    plt.show()
