import pandas as pd
from SampleGenerator import SampleGenerator


# DataFrame Class
# ================

class DataFrame:
    def __init__(self, S: SampleGenerator):
        """
        Define data frames
        :parameter  S: SampleGenerator class element
        :return: --- <class 'NoneType'>
        """
        self.df_files = pd.DataFrame.from_records([s.to_dict() for s in S.files])
        self.df_tests = pd.DataFrame.from_records([s.to_dict() for s in S.tests])
        self.df_commits = pd.DataFrame.from_records([s.to_dict() for s in S.commits])
        self.metrics = pd.DataFrame.from_dict(S.metrics, orient='index')

    def print(self):
        """
        Prints class attributes
        :parameter
        :return: --- <class 'list'>
        """
        print('File Dataframe')
        print(self.df_files)
        print('Test Dataframe')
        print(self.df_tests)
        print('Commit Dataframe')
        print(self.df_commits)
        print('Metrics Dataframe')
        print(self.metrics)

    def write(self):
        """
        Writes obtained data frames to .csv file
        :parameter
        :return: --- <class 'NoneType'>
        """
        self.df_files.to_csv("files.csv", sep='\t')
        self.df_tests.to_csv("tests2.csv", sep='\t')
        self.df_commits.to_csv("commits.csv", sep='\t')
        self.metrics.to_csv("metrics.csv", sep='\t')


# ================
S = SampleGenerator()
D = DataFrame(S)
D.print()
D.write()
