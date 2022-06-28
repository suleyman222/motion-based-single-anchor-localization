import numpy as np

from localization_algorithm import MotionBasedLocalization
from robots.robots import RotatingRobot2D
from robots.two_robot_system import TwoRobotSystem


def determine_r_matrix():
    u = [[1., 0.]] * 25 + [[1., 2.]] * 25 + [[-2, 1]] * 25 + [[-1, -1]] * 50 + [[0, 2]] * 25
    count = 400
    p0 = [-9., -5.]
    v0 = [1, 1]
    dt = .5
    is_noisy = True
    r_std = 1.
    v_std = 0.1
    target_ax_std = 1.5
    target_ay_std = 1.0

    reps = 1000
    q = np.array([[0., 0.], [0., 0.]])
    for i in range(reps):
        # target = ControlledRobot2D(u, dt=dt, init_pos=p0, init_vel=u[0])
        target = RotatingRobot2D(dt=dt, init_pos=p0)
        # target = RandomAccelerationRobot2D(p0, v0, dt, ax_noise=target_ax_std, ay_noise=target_ay_std)
        system = TwoRobotSystem(None, target, is_noisy=is_noisy, r_std=r_std, v_std=v_std)
        loc = MotionBasedLocalization(system, count)
        loc.run()

        if loc.localized:
            error = loc.chosen_measurements[loc.idx_loc:] - system.all_target_positions[loc.idx_loc:]
            e_nx = np.mean(error[:, 0]**2)
            e_nx_ny = np.mean(error[:, 0] * error[:, 1])
            e_ny = np.mean(error[:, 1]**2)
            q += np.array([[e_nx, e_nx_ny], [e_nx_ny, e_ny]])
        else:
            reps -= 1
    q /= reps
    print(f"{q} after {reps} reps")
    return q


if __name__ == '__main__':
    determine_r_matrix()
