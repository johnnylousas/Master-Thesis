import itertools

from CCBP_Algorithm import Test


class Commit:

    def __init__(self, commit_id: str, test_suite: list):
        self.commit_id = commit_id
        #self.arrival_date = arrival_date
        self.failRatio = 0
        self.exeRatio = 0
        self.test_suite = test_suite

    def get_test_nrs(self):
        test_nr = []
        for i in self.test_suite:
            test_nr.append(i.test_id)
        return test_nr

    def get_test_status(self):
        test_status = []
        for i in self.test_suite:
            test_status.append(i.status)
        return test_status

    def print(self):
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))