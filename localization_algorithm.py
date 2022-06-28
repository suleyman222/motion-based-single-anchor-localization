import numpy as np
from abc import ABC, abstractmethod
from robots.two_robot_system import TwoRobotSystem
from utils import util
from scipy import signal
from utils.animator import Animator


class BaseLocalization(ABC):
    def __init__(self, robot_system: TwoRobotSystem, count=50):
        self.robot_system = robot_system
        self.count = count
        self.dt = robot_system.dt
        self.idx_loc = 0
        robot_system.get_r_measurement()

        # Every measurement we get two possible locations for the robot. The initial position is not available
        # through measurements, since the algorithm makes use of change in distance.
        self.measured_positions = np.zeros((count, 2, 2))
        self.chosen_measurements = np.zeros((count, 2))
        self.chosen_measurements[0] = [None, None]
        self.measured_positions[0] = [[None, None], [None, None]]

        self.filtered_dr = []
        self.filtered_r = [self.robot_system.measured_r[0]]

        self.estimated_positions = np.zeros((count, 2))

    def calculate_possible_positions(self):
        prev_rs = self.robot_system.measured_r
        if self.robot_system.is_noisy:
            if len(prev_rs) < 3:
                dr = signal.savgol_filter(deriv=1, x=prev_rs,
                                          window_length=len(prev_rs), polyorder=1, delta=self.dt)[-1]
                r = signal.savgol_filter(deriv=0, x=prev_rs,
                                         window_length=min(len(prev_rs), 50), polyorder=1, delta=self.dt)[-1]
            else:
                dr = signal.savgol_filter(deriv=1, x=prev_rs,
                                          window_length=min(len(prev_rs), 20), polyorder=1, delta=self.dt)[-1]
                r = signal.savgol_filter(deriv=0, x=prev_rs,
                                         window_length=min(len(prev_rs), 20), polyorder=1, delta=self.dt)[-1]
        else:
            dr = (prev_rs[-1] - prev_rs[-2]) / self.dt
            r = prev_rs[-1]

        self.filtered_r.append(r)
        self.filtered_dr.append(dr)

        v = self.robot_system.measured_v[-1]
        s = np.linalg.norm(v)
        alpha = np.arctan2(v[1], v[0])
        theta = np.arccos(util.clamp(dr / s, -1, 1))

        pos1 = [r * np.cos(alpha + theta), r * np.sin(alpha + theta)] + self.robot_system.anchor_robot.pos
        pos2 = [r * np.cos(alpha - theta), r * np.sin(alpha - theta)] + self.robot_system.anchor_robot.pos
        return [pos1, pos2]

    def animate_results(self, title, save=False, plot_error_figures=False):
        target_positions = self.robot_system.all_target_positions.T
        anchor_positions = self.robot_system.all_anchor_positions.T
        estimated_positions = self.estimated_positions.T

        measurements_t = self.measured_positions[:].T
        measurements_reshaped = np.stack((measurements_t[0], measurements_t[1]), axis=1)
        measurements_1 = measurements_reshaped[0]
        measurements_2 = measurements_reshaped[1]

        real_r = np.array(self.robot_system.real_r)
        measured_r = np.array(self.robot_system.measured_r)

        ani = Animator(title, self.count, self.idx_loc, anchor_positions, target_positions, estimated_positions,
                       measurements_1, measurements_2, real_r, measured_r, save, plot_error_figures,
                       np.array(self.filtered_dr) * self.dt, np.linalg.norm(self.robot_system.measured_v, axis=1))
        ani.run()

    @abstractmethod
    def run(self):
        pass


class MotionBasedLocalization(BaseLocalization):
    def __init__(self, robot_system: TwoRobotSystem, count=50, kf=None, known_initial_pos=False):
        super().__init__(robot_system, count)
        self.kf = kf
        self.localized = False
        self.prev_v = robot_system.get_v_measurement()

        if known_initial_pos:
            self.localized = True
            self.idx_loc = 0
            self.estimated_positions[0] = robot_system.target_robot.pos - robot_system.anchor_robot.pos

    def run(self):
        def find_max_similarity(prev_positions, new_positions):
            max_sim = -2
            max_position = None
            for pos in new_positions:
                for prev in prev_positions:
                    sim = util.cos_similarity(pos, prev)
                    if sim > max_sim:
                        max_sim = sim
                        max_position = pos
            return max_position

        for i in range(1, self.count):
            self.robot_system.update()
            self.robot_system.get_r_measurement()
            measured_v = self.robot_system.get_v_measurement()
            measurement1, measurement2 = self.calculate_possible_positions()
            self.measured_positions[i] = [measurement1, measurement2]

            if self.localized:
                # Compare measurements to moving average of previous positions
                window_length = 5
                window = list(range(max(self.idx_loc, i-window_length), i))
                prev_pos = np.mean(self.estimated_positions[window], axis=0)
                closest_measurement = util.closest_to(prev_pos, [measurement1, measurement2])
                self.chosen_measurements[i] = closest_measurement

                if self.kf:
                    self.kf.predict()
                    self.kf.update(np.concatenate((closest_measurement, measured_v)))
                    closest_measurement = [self.kf.x[0], self.kf.x[1]]
                self.estimated_positions[i] = closest_measurement
            else:
                similarity = util.cos_similarity(self.prev_v, measured_v)
                if similarity < .97 and i > 20:
                    self.localized = True
                    prev1 = self.measured_positions[i-1][0]
                    prev2 = self.measured_positions[i-1][1]

                    max_pos = find_max_similarity([prev1, prev2], [measurement1, measurement2])
                    self.estimated_positions[i] = max_pos

                    if self.kf:
                        self.kf.x = np.concatenate((max_pos, measured_v))

                    self.chosen_measurements[i] = max_pos
                    self.idx_loc = i
            self.prev_v = measured_v

        self.robot_system.all_anchor_positions = np.array(self.robot_system.all_anchor_positions)
        self.robot_system.all_target_positions = np.array(self.robot_system.all_target_positions)
