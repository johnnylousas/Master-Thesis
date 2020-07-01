from CCBP_Algorithm import Commit
from operator import attrgetter


class CommitQ:

    def __init__(self):
        self.commitQ = list()

    def add(self, commit: Commit):
        self.commitQ.append(commit)

    def notEmpty(self):
        if self.commitQ:
            return True

    def remove(self):
        return self.commitQ.pop()

    def sortBy(self, failRatio: str, exeRatio: str):
        sorted(self.commitQ, key=attrgetter(failRatio, exeRatio))

    def print(self):
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))