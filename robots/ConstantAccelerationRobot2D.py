import filterpy.common
import numpy as np
from robots.BaseRobot2D import BaseRobot2D
from utils import Util


class ConstantAccelerationRobot2D(BaseRobot2D):
    def __init__(self, init_pos=None, init_vel=None, accel=None, dt=1., noise=False, r_std=0., v_std=0.):
        super().__init__(init_pos=init_pos)

        if accel is None:
            accel = [.1, .1]
        if init_vel is None:
            init_vel = [1., 1.]

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
            # self.last_measured_pos = pos1
            return pos1
        else:
            # self.last_measured_pos = pos2
            return pos2


if __name__ == '__main__':
    pos0 = [3., 2.]
    v0 = [1., 3.]
    acc = [.5, .7]
    dt = 0.1
    r_std = .5
    v_std = .5
    rob = ConstantAccelerationRobot2D(init_pos=pos0, init_vel=v0, accel=acc, dt=dt, r_std=r_std, v_std=v_std, noise=True)

    kf = filterpy.common.kinematic_kf(2, 2, dt, order_by_dim=False)
    kf.x = pos0 + v0 + acc
    kf.R *= (r_std**2 + v_std**2)
    # Low, bcs we are confident in initial position
    kf.P *= 0.001
    kf.Q *= 0
    print(kf)

    count = 100
    estimated_positions = [pos0]
    measured_positions = [pos0]
    for _ in range(count):
        rob.update()
        est_pos = rob.get_position_measurement()
        kf.predict()
        kf.update(est_pos)
        print(kf.x)
        estimated_positions.append([kf.x[0], kf.x[1]])
        measured_positions.append(est_pos)

    # Plot results
    Util.plot_path(np.array(rob.all_positions), np.array(measured_positions), np.array(estimated_positions))

