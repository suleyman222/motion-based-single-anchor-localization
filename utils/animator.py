import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


class Animator:
    def __init__(self, loc, title: str, save=False):
        self.save = save
        self.loc = loc

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(title)
        self.line_anchor, = self.ax.plot([], [], 'm-', ms=10, label="Anchor robot path")
        self.line_actual_target, = self.ax.plot([], [], 'b-', ms=10, label="Actual target path")
        self.line_chosen, = self.ax.plot([], [], 'r-', ms=10, label="Estimated target path")
        self.line_measured, = self.ax.plot([], [], 'g--', ms=10, label="Measured target path")

        axamp = plt.axes([0.25, .03, 0.50, 0.02])
        self.slider = Slider(axamp, 'Timestep', 0, self.loc.count, valinit=0, valstep=1)
        self.is_manual = False

        self.anchor_x = self.loc.robot_system.all_anchor_positions[:, 0]
        self.anchor_y = self.loc.robot_system.all_anchor_positions[:, 1]
        self.target_x = self.loc.robot_system.all_target_positions[:, 0]
        self.target_y = self.loc.robot_system.all_target_positions[:, 1]
        self.chosen_x = self.loc.estimated_positions[1:, 0]
        self.chosen_y = self.loc.estimated_positions[1:, 1]
        self.measured_x = self.loc.measured_positions[1:, 0]
        self.measured_y = self.loc.measured_positions[1:, 1]

    def run(self):
        all_x = np.concatenate((self.anchor_x, self.target_x, self.chosen_x, self.measured_x))
        all_y = np.concatenate((self.anchor_y, self.target_y, self.chosen_y, self.measured_y))
        self.ax.set_xlim(np.min(all_x), np.max(all_x))
        self.ax.set_ylim(np.amin(all_y), np.amax(all_y))
        self.ax.legend()

        # call update function on slider value change
        self.slider.on_changed(self.update_slider)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        ani = matplotlib.animation.FuncAnimation(self.fig, self.animate, frames=self.loc.count, interval=50,
                                                 save_count=self.loc.count)
        if self.save:
            ani.save('ani.gif', 'pillow')
        plt.show()

    def animate(self, frame):
        if self.is_manual:
            return self.line_anchor, self.line_actual_target, self.line_chosen, self.line_measured

        # Calls update due to change
        self.slider.set_val(frame)
        self.is_manual = False
        return self.line_anchor, self.line_actual_target, self.line_chosen, self.line_measured

    def update_slider(self, val):
        self.is_manual = True
        self.update(val)

    def update(self, val):
        self.line_anchor.set_data(self.anchor_x[:val], self.anchor_y[:val])
        self.line_actual_target.set_data(self.target_x[:val], self.target_y[:val])
        self.line_chosen.set_data(self.chosen_x[:val], self.chosen_y[:val])
        self.line_measured.set_data(self.measured_x[:val], self.measured_y[:val])
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        # Check where the click happened
        (xm, ym), (xM, yM) = self.slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = unpause
            self.is_manual = False
