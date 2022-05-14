import math
import filterpy.common
import filterpy.kalman.kalman_filter as kalman_filter
import numpy as np
from filterpy.common import Saver

from matplotlib import pyplot as plt
from numpy.random import randn

from robots.BaseRobot2D import BaseRobot2D
from robots.ConstantAccelerationRobot2D import ConstantAccelerationRobot2D

# Ideas to check out:
# - a changing value of dt
# - different data rates of sensors


class PositionTracking:
    def __init__(self, P, Q, R, dt=1., robot: BaseRobot2D = ConstantAccelerationRobot2D(), init_pos=None, init_v=None, init_a=None, count=50):
        if init_pos is None:
            init_pos = [robot.x, robot.y]
        if init_v is None:
            init_v = [0., 0.]
        if init_a is None:
            init_a = [1., 1.]

        # self.kf = kalman_filter.KalmanFilter(dim_x=6, dim_z=2)
        # self.kf.x = [init_pos[0], init_pos[1], init_v[0], init_v[1], init_a[0], init_a[1]]
        # self.kf.F = np.array([[1., 0., dt, 0., .5*dt*dt, 0.],
        #                       [0., 1., 0., dt, 0., .5*dt*dt],
        #                       [0., 0., 1., 0., dt, 0.],
        #                       [0., 0., 0., 1., 0., dt],
        #                       [0., 0., 0., 0., 1., 0.],
        #                       [0., 0., 0., 0., 0., 1.]])
        # self.kf.H = [1., 1., 0., 0., 0., 0.]
        # self.kf.R *= R  # measurement uncertainty
        # if np.isscalar(P):
        #     self.kf.P *= P  # covariance matrix
        # else:
        #     self.kf.P[:] = P  # [:] makes deep copy
        # if np.isscalar(Q):
        #     self.kf.Q = filterpy.common.Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        # else:
        #     self.kf.Q[:] = Q

        # Constant acceleration model (2 dimensional, 2nd order)
        self.kf = filterpy.common.kinematic_kf(2, 2, dt, order_by_dim=False)
        self.kf.P *= P
        self.kf.R *= R
        self.kf.Q *= Q

        self.robot = robot
        self.count = count
        self.pos = np.array(init_pos)
        self.distance = np.sqrt(self.pos.dot(self.pos))

        self.dt = dt

    def runWithoutKalman(self):
        pos_before_kf = np.array([])
        for _ in range(self.count):
            pos = self.get_position_measurement()
            np.append(pos_before_kf, pos)

        # Plot x coordinate
        plt.plot([pos[0] for pos in pos_before_kf], label='x_predic')
        plt.plot(self.robot.xs, label='x_actual')

        # Plot y coordinate
        plt.plot([pos[1] for pos in pos_before_kf], label='y_predic')
        plt.plot(self.robot.ys, label='y_actual')

        plt.legend()
        plt.show()

    def run(self):
        sav = Saver(self.kf)
        pos_before_kf = np.array([])
        for _ in range(self.count):
            pos = self.get_position_measurement()
            np.append(pos_before_kf, pos)
            self.kf.predict()
            self.kf.update(pos)
            sav.save()
        sav.to_array()
        # print(pos_before_kf[:][1])
        # plt.plot(pos_before_kf[:, 0], pos_before_kf[:, 1])
        plt.plot(sav.x[:, 0], sav.x[:, 1], label='prediction')
        plt.plot(self.robot.xs, self.robot.ys, label='actual pos')
        plt.legend()
        plt.show()


    def simulate_acc_system(self, R, Q, count):
        robot = ConstantAccelerationRobot2D(acc_noise=Q)
        zs = []
        xs = []
        for i in range(count):
            x = robot.update()
            z = sense(x, R)
            xs.append(x)
            zs.append(z)
        return np.asarray(xs), zs

    def compute_data(self, z_var, process_var, count=1, dt=1.):
        "returns track, measurements 1D ndarrays"
        x, y, vx, vy, ax, ay = 0., 0., 0., 0., 1., 1.
        z_std = math.sqrt(z_var)
        p_std = math.sqrt(process_var)
        xs, ys, zs = [], [], []
        for _ in range(count):
            # acc =
            v = vx + (randn() * p_std)
            x += v * dt
            xs.append(x)
            zs.append(x + randn() * z_std)
        return np.array(xs), np.array(zs)

    def get_position_measurement(self):
        x, y, vx, vy, ax, ay = self.robot.update()
        # return np.array([x, y])

        r = np.sqrt(x**2 + y**2)
        dr = (r - self.distance) / self.dt
        self.distance = r

        v = np.array([vx, vy])
        s = np.sqrt(v.dot(v))

        alpha = np.arctan(vy / vx)
        theta = np.arccos(dr / s)

        pos1 = np.array([r * np.cos(alpha + theta), r * np.sin(alpha + theta)])
        pos2 = np.array([r * np.cos(alpha - theta), r * np.sin(alpha - theta)])
        # return pos1

        if np.linalg.norm(pos1-self.pos) < np.linalg.norm(pos2-self.pos):
            self.pos = pos1
            return pos1
        else:
            self.pos = pos2
            return pos2


# R, Q = 6., 0.02


def simulate_acc_system(R, Q, count):
    obj = ConstantAccelerationRobot2D(acc_noise=0)
    zs = []
    xs = []
    for i in range(count):
        x = obj.update()
        z = sense(x, R)
        xs.append(x)
        zs.append(z)
    return np.asarray(xs), zs


def sense(x, noise_scale=1.):
    return x[0] + randn()*noise_scale


if __name__ == '__main__':
    # xs, zs = simulate_acc_system(R=R, Q=Q, count=40)
    # plt.plot(xs[:, 0])
    # plt.plot(zs)
    # plt.show()

    rob = ConstantAccelerationRobot2D(3, 1)

    pt = PositionTracking(2.5, 2, 2, 2, robot=rob)
    # pt.run()

    noiseless = PositionTracking(P=1, Q=1, R=1, robot=rob)
    noiseless.runWithoutKalman()
