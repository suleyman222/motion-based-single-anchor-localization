from abc import ABC, abstractmethod


class BaseRobot2D(ABC):
    def __init__(self, init_pos=None):
        if init_pos is None:
            init_pos = [0., 0.]

        self.init_pos = init_pos

    @abstractmethod
    def update(self):
        pass
    