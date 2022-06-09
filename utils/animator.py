import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation


class Animator:
    def __init__(self, title: str, count, idx_loc, anchor_pos, target_pos, estimated_target_pos, measured_target_pos_1,
                 measured_target_pos_2, real_r=None, measured_r=None, save=False,
                 plot_error_figures=False, used_dr=None, speed=None):
        self.speed = speed
        self.used_dr = used_dr
        self.plot_error_figures = plot_error_figures
        self.save = save
        self.idx_loc = idx_loc
        self.count = count
        self.measured_r = measured_r
        self.real_r = real_r

        self.anchor_pos = anchor_pos
        self.target_pos = target_pos
        self.estimated_target_pos = estimated_target_pos
        self.measured_target_pos_1 = measured_target_pos_1
        self.measured_target_pos_2 = measured_target_pos_2

        if plot_error_figures:
            self._initialize_error_figures()
        else:
            self.fig = plt.figure()
            self.ax_animation = self.fig.add_subplot(111)

        self.line_anchor, = self.ax_animation.plot(anchor_pos[0], anchor_pos[1], 'm-',
                                                   ms=10, label="Anchor robot path")

        self.line_measured1, = self.ax_animation.plot(measured_target_pos_1[0], measured_target_pos_1[1],
                                                      'g--', ms=10, label="Measured target path 1")
        self.line_measured2, = self.ax_animation.plot(measured_target_pos_2[0], measured_target_pos_2[1],
                                                      'y--', ms=10, label="Measured target path 2")
        self.line_estimated, = self.ax_animation.plot(estimated_target_pos[0], estimated_target_pos[1],
                                                      'r-', ms=10, label="Estimated target path")
        self.line_actual_target, = self.ax_animation.plot(target_pos[0], target_pos[1], 'b-',
                                                          ms=10, label="Actual target path")

        self.fig.suptitle(title)
        self.ax_animation.legend()
        self.ax_animation.set_xlabel("X coordinate")
        self.ax_animation.set_ylabel("Y coordinate")

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        self.slider = Slider(axamp, 'Timestep', 0, self.count, valinit=0, valstep=1)
        self.is_manual = False

    def run(self):
        # call update function on slider value change
        self.slider.on_changed(self._update_slider)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        ani = matplotlib.animation.FuncAnimation(self.fig, self._animate, frames=self.count, interval=50,
                                                 save_count=self.count)
        if self.save:
            ani.save('ani.gif', 'pillow')
        plt.show()

    def _animate(self, frame):
        if self.is_manual:
            return self.line_anchor, self.line_actual_target, self.line_estimated,\
                   self.line_measured1, self.line_measured2

        # Calls update due to change
        self.slider.set_val(frame)
        self.is_manual = False
        return self.line_anchor, self.line_actual_target, self.line_estimated, self.line_measured1, self.line_measured2

    def _update_slider(self, val):
        self.is_manual = True
        self._update(val)

    def _update(self, val):
        self.line_anchor.set_data(self.anchor_pos[0][:val], self.anchor_pos[1][:val])
        self.line_actual_target.set_data(self.target_pos[0][:val], self.target_pos[1][:val])
        self.line_estimated.set_data(self.estimated_target_pos[0][self.idx_loc:val],
                                     self.estimated_target_pos[1][self.idx_loc:val])
        self.line_measured1.set_data(self.measured_target_pos_1[0][:val], self.measured_target_pos_1[1][:val])
        self.line_measured2.set_data(self.measured_target_pos_2[0][:val], self.measured_target_pos_2[1][:val])

        if self.plot_error_figures:
            self.line_r_error.set_data(range(val), self.speed[:val])
            self.line_dr_error.set_data(range(val), self.dr_error[:val])
            self.line_dr.set_data(range(val), self.dr[:val])
            self.line_filtered_dr_error.set_data(range(val), self.filtered_dr_error[:val])
            self.line_dr_measured.set_data(range(val), self.measured_dr[:val])
            if val >= self.idx_loc:
                self.line_pos_error.set_data(range(self.idx_loc, val), self.pos_err[:val - self.idx_loc])
            else:
                self.line_pos_error.set_data([], [])

        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        # Check where the click happened
        (xm, ym), (xM, yM) = self.slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            self.is_manual = False

    def _initialize_error_figures(self):
        fig, axs = plt.subplots(2, 2)
        self.fig = fig
        self.ax_animation = axs[0][0]

        self.speed = self.speed
        self.dr = np.insert(np.diff(self.real_r), 0, None)
        self.measured_dr = np.insert(np.diff(self.measured_r), 0, None)
        self.dr_error = np.insert(np.abs(self.dr[1:] - self.measured_dr[1:]), 0, None)
        self.filtered_dr_error = np.insert(np.abs(self.dr[1:] - self.used_dr), 0, None)
        self.pos_err = np.linalg.norm(self.target_pos.T[self.idx_loc:] - self.estimated_target_pos.T[self.idx_loc:],
                                      axis=1)

        self.line_r_error, = axs[1][0].plot(self.speed, label="Noise in r")
        self.line_pos_error, = axs[0][1].plot(range(self.idx_loc, self.count), self.pos_err, label="$||p - \hat{p}||$")
        self.line_dr_error, = axs[1][1].plot(range(self.count), self.dr_error, label="Noise in Forward Difference")
        self.line_filtered_dr_error, = axs[1][1].plot(range(self.count), self.filtered_dr_error, label="Noise in SG")
        self.line_dr_measured, = axs[1][1].plot(range(self.count), self.measured_dr, label="Forward Difference")
        self.line_dr, = axs[1][1].plot(range(self.count), self.dr, label="Real dr")
        axs[1][1].plot(range(self.count), np.insert(self.used_dr, 0, None), label="Savitzky-Golay filtered")

        axs[0][1].set_title("Error in position estimate $||p - \hat{p}||$")
        axs[0][1].set_xlim(0)
        axs[0][1].set_xlabel("Time step")
        axs[1][0].set_title("Speed")
        axs[1][0].set_xlim(0)
        axs[1][0].set_xlabel("Time step")
        axs[1][1].legend()
        axs[1][1].set_title("Plots of change of r (dr)")
        axs[1][1].set_xlim(0)
        axs[1][1].set_xlabel("Time step")
