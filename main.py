import filterpy.common

from position_tracking import PositionTracking
from robots.constant_acceleration_robot import ConstantAccelerationRobot2D

if __name__ == '__main__':
    pos0 = [3., 2.]
    v0 = [1., 3.]
    acc = [.5, .7]
    # Lower dt -> more noise (why?)
    dt = 0.3
    r_std = .01
    v_std = .01
    rob = ConstantAccelerationRobot2D(
        init_pos=pos0, init_vel=v0, accel=acc, dt=dt, r_std=r_std, v_std=v_std, noise=True)

    # Constant acceleration model (2 dimensional, 2nd order)
    kf = filterpy.common.kinematic_kf(2, 2, dt, order_by_dim=False)
    # x = [x, y, vx, vy, ax, ay]
    kf.x = pos0 + v0 + acc
    kf.R *= (r_std**2 + v_std**2)
    # Low, bcs we are confident in initial position
    kf.P *= 0.001
    kf.Q *= 0
    # print(kf)

    pt = PositionTracking(kf, rob, count=100)
    pt.run()
