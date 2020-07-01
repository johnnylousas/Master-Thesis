from CCBP_Algorithm import Commit


class Resources:

    def __init__(self, size: int):
        self.resources = list()
        self.size = size
        self.available = True

    def available(self):
        return self.available

    def release(self):
        self.available = True

    def allocate(self, commit: Commit):
        self.resources.append(commit)
        self.available = False

    def print(self):
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))