from abc import abstractmethod
import random


class Group:

    def __init__(self):
        groups = ['A', 'B', 'C', 'D']
        self.group = random.sample(groups, k=1)[0]

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def print(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass


