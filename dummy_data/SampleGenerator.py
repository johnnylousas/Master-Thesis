import random
import pandas as pd
from faker import Faker
from File import File
from Test import Test
from Commit import Commit
from dummy_data import tools
from numpy import mean


# SampleGenerator Class
# ================


class SampleGenerator:

    def __init__(self, nr_files: int = 10, nr_tests: int = 10, nr_commits: int = 10, nr_developers: int = 4):
        """
        Define data frames from given parameters
        :parameter  nr_records: number of files in database
                    nr_tests: number of tests that can be applied
                    nr_commits: number of commits made
                    nr_developers: team size
        :return: --- <class 'NoneType'>
        """

        self.files = list()
        self.tests = list()
        self.commits = list()

        for i in range(nr_files):
            self.files.append(File())
        for i in range(nr_tests):
            self.tests.append(Test())

        self.commits = self.commit(nr_commits=nr_commits, team_size=nr_developers)
        self.metrics = self.retrieve_metrics(nr_files=nr_files, nr_tests=nr_tests, nr_commits=nr_commits,
                                             nr_developers=nr_developers)

    def commit(self, nr_commits: int, team_size: int):
        commits = list()
        authors = list()
        fake = Faker()

        # generate n timestamps
        t = tools.timestamps(nr_commits)

        # fill in random authors, according to team size
        for x in range(team_size):
            authors.append(fake.name())

        # Create Commit class element
        for i in range(nr_commits):
            commits.append(Commit(author=random.sample(authors, k=1), timestamp=t[i], files=self.check_prevalence()))

        return commits

    def check_prevalence(self):
        modified_files = list()
        for i in self.files:
            if random.randint(0, 100) < i.get_prevalence():
                modified_files.append(i)
        return modified_files

    def retrieve_metrics(self, nr_files, nr_tests, nr_commits, nr_developers):

        nr_files_modified_per_person = []
        [nr_files_modified_per_person.append(i.get_nr_modified_files()) for i in self.commits]

        metrics = {
            'nr_files': nr_files,
            'nr_tests': nr_tests,
            'nr_commits': nr_commits,
            'team_size': nr_developers,
            'avg_nr_files_modified_per_person': mean(nr_files_modified_per_person)
        }
        return metrics

    def print(self):
        """
        Prints class attributes
        :parameter
        :return: --- <class 'list'>
        """
        print('File list')
        [i.print() for i in self.files]
        print('Test list')
        [i.print() for i in self.tests]
        print('Commit list')
        [i.print() for i in self.commits]

# ================
