from CCBP_Algorithm.Commit import Commit
from CCBP_Algorithm.Test import Test
from CCBP_Algorithm.Resources import Resources
from CCBP_Algorithm.CommitQ import CommitQ

import pandas as pd
from collections import defaultdict


def commitsSinceLastExecution(test_id: str):
    pass


def commitsSinceLastFailure(test_id: str):
    for r in range(len(commit_list), 0):
        test_ids = commit_list[r].get_test_nrs()
        commit_status = commit_list[r].get_test_status()
        for i in range(len(test_ids)):
            if test_id == test_ids[i]:
                if commit_status[i] == "FAILED":
                    return r


def createCommit(df: pd.DataFrame, commit_id: int):
    df_copy = df.copy()
    df_commit1 = df_copy.loc[df['ChangeRequest '] == commit_id]
    test_nr = df_commit1['test_id'].tolist()
    test_status = df_commit1['Status'].tolist()
    launch_time = df_commit1['LaunchTime'].tolist()
    exe_time = df_commit1['ExecutionTime '].tolist()

    test_suites = []
    for i in range(len(test_nr)):
        test_suites.append(Test(test_id=str(test_nr[i]), status=test_status[i], launch_time=launch_time[i],
                                exe_time=exe_time[i]))

    return Commit(commit_id=commit_id, test_suite=test_suites)


class CCBP:

    def __init__(self, failWindowSize: int, exeWindowSize: int):
        self.failWindowSize = failWindowSize
        self.exeWindowSize = exeWindowSize
        self.resources = Resources(size=1)
        self.commitQ = CommitQ()

    def onCommitArrival(self, cmt: Commit):
        self.commitQ.add(cmt)
        if self.resources.available():
            self.prioritize()

    def onCommitTestEnding(self):
        self.resources.release()
        if self.commitQ.notEmpty():
            self.prioritize()

    def prioritize(self):
        for i in self.commitQ.commitQ:
            self.updateCommitInformation(i)
        self.commitQ.sortBy("failRatio", "exeRatio")
        cmt = self.commitQ.remove()
        self.resources.allocate(cmt)

    def updateCommitInformation(self, cmt: Commit):
        failCounter = exeCounter = numTests = 0
        for i in cmt.test_suite.test_id:
            numTests += 1
            if commitsSinceLastFailure(str(i)) <= self.failWindowSize:
                failCounter += 1
            if commitsSinceLastExecution(str(i)) <= self.exeWindowSize:
                exeCounter += 1
        cmt.failRatio = failCounter / numTests
        cmt.exeRatio = exeCounter / numTests


if __name__ == '__main__':
    # read data
    data_path = '../data_csv/'
    df = pd.read_csv(data_path + 'GooglePostCommit.csv', nrows=1000)

    # convert Test Suite string into simple test_id
    test_list = df.iloc[:, 0]
    temp = defaultdict(lambda: len(temp))
    test_id = [temp[ele] for ele in test_list]

    # remove first column
    df.insert(loc=1, column='test_id', value=test_id)
    df = df.loc[:, df.columns != 'TestSuite']

    # choose tests per commit
    # print(df.loc[df['ChangeRequest '] == 0])
    change_requests = df['ChangeRequest '].unique()

    print('**DATASET INFORMATION**')
    print('total nr of commits  ->  ' + str(len(change_requests)))
    print('nr of unique test suites  ->  ' + str(len(df['test_id'].unique())))
    print('total nr of test executions  ->  ' + str(len(df['test_id'])))
    print('***********************')

    # Create commits
    commit_list = []
    for commit in change_requests:
        commit_list.append(createCommit(df, commit_id=commit))

    print(commitsSinceLastFailure(test_id=34))