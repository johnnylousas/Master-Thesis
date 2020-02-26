import random
import pandas as pd
from faker import Faker
import tools
import datetime


# Data Generator Class
# ================


class DataGenerator:

    def __init__(self, nr_records: int = 10, nr_tests: int = 10, nr_commits: int = 10, nr_developers: int = 4):
        """
        Define data frames from given parameters
        :parameter  nr_records: number of files in database
                    nr_tests: number of tests that can be applied
                    nr_commits: number of commits made
                    nr_developers: team size
        :return: --- <class 'NoneType'>
        """
        self.nr_records = nr_records
        self.nr_tests = nr_tests
        self.nr_commits = nr_commits
        self.nr_developers = nr_developers

        self.fieldnames_src = ['id', 'prevalence', 'stability']
        self.fieldnames_test = ['id', 'time_to_run', 'dependence']
        self.fieldnames_commits = ['commit_id', 'author', 'timestamp', 'modified_files', 'broken_files']
        self.fieldnames_run_tests = ['commit_id', 'test_id', 'timestamp', 'test_status']

        self.df_src = pd.DataFrame(self.create_rows_src())
        self.df_test = pd.DataFrame(self.create_rows_test())
        self.df_cmt = pd.DataFrame(self.create_rows_cmt())
        self.df_run_test = pd.DataFrame(self.create_rows_run_test_hist())

    def create_rows_src(self):
        """
        Generates source list filled with faked data. (Seeded - always same outcome for simplicity)
        :parameter  s: column length
        :return: --- <class 'list'>
        """
        random.seed(2)
        fake = Faker()
        Faker.seed(2)
        output = [{
            "id": fake.file_name(category=None, extension=None),
            "prevalence": random.randint(0, 100),
            "stability": random.randint(90, 100)} for x in range(self.nr_records)]
        return output

    def create_rows_test(self):
        """
        Generates test list filled with faked data. (Seeded - always same outcome for simplicity)
        :parameter
        :return: --- <class 'list'>
        """
        random.seed(2)
        fake = Faker()
        Faker.seed(2)
        output = [{
            "id": fake.file_name(category=None, extension='test'),
            "time_to_run": random.randint(0, 200),
            "dependence": random.sample(self.df_src['id'].tolist(), k=random.randint(1, 4))} for x in
            range(self.nr_tests)]
        return output

    def create_rows_cmt(self):
        """
        Generates commit list filled with faked data. (Not Seeded )
        :parameter
        :return: --- <class 'list'>
        """
        random.seed()
        fake = Faker()
        t = tools.timestamps(self.nr_commits)
        mod = []
        broken = []
        dev_name = []
        for x in range(self.nr_records):
            m, b = self.check_mod()
            mod.append(m)
            broken.append(b)
        for x in range(self.nr_developers):
            dev_name.append(fake.name())

        output = [{
            "commit_id": fake.ean8(),
            "author": random.sample(dev_name, k=1),
            "timestamp": t[x],
            "modified_files": mod[x],
            "Broken_files": broken[x]} for x in range(self.nr_records)]
        return output

    def check_mod(self):
        """
        Based on prevalence and stability from source table, generates modified and broken files.
        :parameter
        :return: --- <class 'list', class 'list'>
        """
        df_mod = self.df_src.copy()
        id = self.df_src['id'].tolist()
        prev = self.df_src['prevalence'].tolist()

        modified_files = []
        broken_files = []
        for i in range(len(prev)):
            survival_rate = tools.random_gen(0, 100)
            if prev[i] >= survival_rate:
                modified_files.append(id[i])
            else:
                df_mod.drop(i, inplace=True)
        stab = df_mod['stability'].tolist()
        id2 = df_mod['id'].tolist()
        for i in range(len(stab)):
            survival_rate = tools.random_gen(0, 100)
            if stab[i] < survival_rate:
                broken_files.append(id2[i])
        return modified_files, broken_files

    def create_rows_run_test_hist(self):
        """
        Generates list of already run tests, applying each test to each commit (first approach)
        :parameter
        :return: --- <class 'list'>
        """
        test_list = self.df_test['id'].to_list()
        duration = self.df_test['time_to_run'].to_list()
        print(duration)
        for i in range(len(duration)-1):
            duration[i+1] += duration[i]
        print(duration)
        test_dependence = self.df_test['dependence'].to_list()

        commit_id = self.df_cmt['commit_id'].to_list()
        dates_list = [datetime.datetime.strptime(date, '%d/%m/%y %H:%M') for date in self.df_cmt['timestamp'].to_list()]
        broken_files = self.df_cmt['Broken_files'].to_list()

        output = [{
            "id": commit_id[i],
            "test_id": test_list[x],
            "timestamp": dates_list[i] + datetime.timedelta(minutes=duration[x]),
            "test_status": self.apply_test(test_dependence[x], broken_files[i])
        } for i in range(len(commit_id)) for x in range(self.nr_tests)]
        return output

    def check_dep(self, files: list, test_dep: list):
        """
        Checks if some test dependency (file) is part of the list of the modified files in a given commit
        :parameter files <class 'list'>
                   test_dep <class 'list'>
        :return: --- <class 'bool'>
        """
        for i in test_dep:
            if files.__contains__(i):
                return True
            else:
                return False

    def apply_test(self, dependencies: list, broken_files: list):
        """
        Applies a test to a commit, if one dependency of the test belongs to the list of broken files, then the test
        fails when applied to that commit.
        :parameter dependencies <class 'list'>
                   broken_files <class 'list'>
        :return: --- <class 'string'>
        """
        P, F = ['Pass', 'Fail']
        if self.check_dep(broken_files, dependencies):
            return F
        else:
            return P

    def print(self):
        """
        Prints class data frames
        :parameter
        :return: --- <class 'NoneType'>
        """
        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(self.df_src)
            print(self.df_test)
            print(self.df_cmt)
            print(self.df_run_test)

    def write(self):
        """
        Writes obtained data frames to .csv file
        :parameter
        :return: --- <class 'NoneType'>
        """
        self.df_src.to_csv("src.csv", sep='\t')
        self.df_test.to_csv("test.csv", sep='\t')
        self.df_cmt.to_csv("cmt.csv", sep='\t')
        self.df_run_test.to_csv("run_test_history.csv", sep='\t')


# ================
# Create class elemente Data Generator
D = DataGenerator()

# Write new .csv file
# D.write()
