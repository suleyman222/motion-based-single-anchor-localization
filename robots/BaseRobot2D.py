from abc import ABC, abstractmethod


class BaseRobot2D(ABC):
    def __init__(self, init_pos=None, dt=1.):
        if init_pos is None:
            init_pos = [0., 0.]

        self.init_pos = init_pos
        self.dt = dt

    @abstractmethod
    def update(self):
        pass
    