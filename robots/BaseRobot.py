from abc import ABC, abstractmethod


class Robot(ABC):
    def __init__(self, x0=0, y0=0):
        self.x = x0
        self.y = y0

    @abstractmethod
    def update(self):
        pass
    