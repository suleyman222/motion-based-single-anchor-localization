import numpy as np
from typing import Optional
from robots.robots import ConstantAccelerationRobot2D, BaseRobot2D


class TwoRobotSystem:
    def __init__(self, anchor_robot: Optional[BaseRobot2D], target_robot: BaseRobot2D, is_noisy=False, rssi_noise=False,
                 r_std=0., v_std=0.):
        if anchor_robot is None:
            anchor_robot = ConstantAccelerationRobot2D([0., 0.], [0., 0.], [0., 0.], dt=target_robot.dt)

        self.v_std = v_std
        self.r_std = r_std
        self.is_noisy = is_noisy
        self.rssi_noise = rssi_noise
        self.anchor_robot = anchor_robot
        self.target_robot = target_robot
        self.dt = target_robot.dt

        self.all_anchor_positions = [anchor_robot.pos]
        self.all_target_positions = [target_robot.pos]

        # Measurements
        self.real_r = []
        self.measured_r = []
        self.measured_v = []

        self.prev_rssis = [-60]

        if anchor_robot.dt != target_robot.dt:
            print("Target and anchor dt are different!")

    def update(self):
        self.target_robot.update()
        self.anchor_robot.update()
        self.all_anchor_positions.append(self.anchor_robot.pos)
        self.all_target_positions.append(self.target_robot.pos)

    def get_v_measurement(self):
        v_tracked_robot = self.target_robot.vel
        v_anchor_robot = self.anchor_robot.vel
        v = v_tracked_robot - v_anchor_robot

        if self.is_noisy:
            v = v + np.random.normal(0, self.v_std)
        self.measured_v.append(v)
        return v

    def get_r_measurement(self):
        r = np.linalg.norm(self.target_robot.pos - self.anchor_robot.pos)
        self.real_r.append(r)

        if self.is_noisy:
            if self.rssi_noise:
                path_loss_exp = 1.8
                p_0 = -52  # [dBm]
                sigma_rssi = 5.8  # [dB]
                p_ij = p_0 - 10 * path_loss_exp * np.log10(r)

                noisy_rssi = p_ij + np.random.normal(0, sigma_rssi)
                self.prev_rssis.append(noisy_rssi)

                # Moving average of RSSI
                window_size = 10
                averaged_rssi = np.mean(self.prev_rssis[len(self.prev_rssis) - window_size:])
                averaged_distance = self.rssi_to_r(averaged_rssi)
                r = averaged_distance
            else:
                r += np.random.normal(0, self.r_std)
        self.measured_r.append(r)
        return r

    @staticmethod
    def rssi_to_r(rssi):
        p_0 = -52
        path_loss_exp = 1.8
        return 10**((p_0 - rssi) / (10 * path_loss_exp))
