import numpy as np
from matplotlib import pyplot as plt
from robots.BaseRobot import Robot
import matplotlib.axes._axes as axes
import matplotlib.figure as figure


class ConstantAccelerationRobot(Robot):
    def __init__(self, x0=0., y0=0., vx0=1., vy0=1., ax0=0.1, ay0=0.1, acc_noise=.1, dt=1.):
        super().__init__()
        self.x = x0
        self.y = y0
        self.vx = vx0
        self.vy = vy0
        self.ax = ax0
        self.ay = ay0
        self.acc_noise_scale = acc_noise
        self.dt = dt
        self.xs = [x0]
        self.ys = [y0]
        self.distance = np.sqrt(self.x**2 + self.y**2)

    def update(self):
        self.ax += self.acc_noise_scale  # * self.dt
        self.ay += self.acc_noise_scale  # * self.dt
        self.vx += self.ax * self.dt
        self.vy += self.ay * self.dt
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        self.xs.append(self.x)
        self.ys.append(self.y)
        return self.x, self.y, self.vx, self.vy, self.ax, self.ay

    def get_position_measurement(self):
        r = np.sqrt(self.x**2 + self.y**2)
        dr = (r - self.distance) / self.dt
        self.distance = r

        # v = np.array([self.vx, self.vy])
        s = np.sqrt(self.vx**2 + self.vy**2)

        alpha = np.arctan(self.vy / self.vx)
        theta = np.arccos(dr / s)

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])

        if np.linalg.norm(pos1-[self.x, self.y]) < np.linalg.norm(pos2-[self.x, self.y]):
            return pos1
        else:
            return pos2


if __name__ == '__main__':
    x0 = 1
    y0 = 2
    count = 100

    rob = ConstantAccelerationRobot(x0=x0, y0=y0, vx0=1, vy0=3, acc_noise=0, ax0=.5, ay0=.6, dt=0.1)
    initial_est_pos = rob.get_position_measurement()
    xs_est, ys_est = [x0], [y0]
    for _ in range(count):
        rob.update()
        est_pos = rob.get_position_measurement()
        xs_est.append(est_pos[0])
        ys_est.append(est_pos[1])

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2) # type:figure.Figure, ((axes.Axes, axes.Axes,), (axes.Axes, axes.Axes,))
    plt.subplots_adjust(hspace=0.7)
    plt.suptitle('Localization without noise, known initial position, constant acceleration')

    # Plot X coordinates
    ax1.plot(rob.xs, label='real x')
    ax1.plot(xs_est, label='estimated x')
    ax1.set_title('X position')
    ax1.set(ylabel='x coordinate')
    print(np.sum(np.array(rob.xs) - np.array(xs_est)))

    ax2.plot(rob.xs, label='real x')
    ax2.plot(xs_est, label='estimated x')
    ax2.set_title('X position (zoomed in)')
    ax2.set_xlim(20, 21)
    ax2.set_ylim(4, 4.3)
    ax2.legend()

    # Plot Y coordinates
    ax3.plot(rob.ys, label='real y', color='red')
    ax3.plot(ys_est, label='estimated y', color='green')
    ax3.set_title('Y position')
    ax3.set(xlabel='time', ylabel='y coordinate')
    print(np.sum(np.array(rob.ys) - np.array(ys_est)))

    ax4.plot(rob.ys, label='real y', color='red')
    ax4.plot(ys_est, label='estimated y', color='green')
    ax4.set_title('Y position (zoomed in)')
    ax4.set(xlabel='time')
    ax4.legend()
    ax4.set_xlim(16, 16.5)
    ax4.set_ylim(7.5, 8)

    plt.show()
