import random
from faker import Faker
import pandas as pd

# File Class
# ================


class File:

    def __init__(self):
        """
        Initializes a File belonging to the repository.
        Each file is associated to a group with the following characteristics:
        Group A: Set Files that have a very low nr of changes(5-10%) and have impact in a lot of tests(50-100%), but
        rarely cause test breaks;
        Group B: Set of files that have a very low nr of changes (5-10%) and have impact in a small amount of tests,
        with randomized break rates;
        Group C: Set of files with frequent change rate (50-100%) and impact small amount of tests(5-20%), with
        randomized break rates;
        Group D: Set of files with random change rate and impact in random amount of tests, with random break rates.
        :parameter  id: filename
                    group: file group
                    prevalence: rate of changes a file suffers
                    stability: chance of file being broke by test case
                    impact: amount of tests each file impacts
        :return: --- <class 'NoneType'>
        """
        fake = Faker()

        groups = ['A', 'B', 'C', 'D']

        self.id = fake.file_name(category=None, extension=None)
        self.group = random.sample(groups, k=1)[0]

        self.prevalence = self.prevalence()
        self.stability = self.stability()
        self.impact = self.impact()

    def prevalence(self):
        """
        Retrieves prevalence according to group.
        :parameter
        :return: --- <class 'int'>
        """
        if self.group.__eq__('A') or self.group.__eq__('B'):
            return random.randint(5, 10)
        elif self.group.__eq__('C'):
            return random.randint(50, 100)
        elif self.group.__eq__('D'):
            return random.randint(0, 100)

    def stability(self):
        """
        Retrieves stability according to group.
        :parameter
        :return: --- <class 'int'>
        """
        if self.group.__eq__('A'):
            return random.randint(95, 100)
        elif self.group.__eq__('B'):
            return random.randint(0, 100)
        elif self.group.__eq__('C'):
            return random.randint(0, 100)
        elif self.group.__eq__('D'):
            return random.randint(0, 100)

    def impact(self):
        """
        Retrieves impact according to group.
        :parameter
        :return: --- <class 'int'>
        """
        if self.group.__eq__('A'):
            return random.randint(50, 100)
        elif self.group.__eq__('B'):
            return random.randint(5, 10)
        elif self.group.__eq__('C'):
            return random.randint(5, 10)
        elif self.group.__eq__('D'):
            return random.randint(0, 100)

    def __repr__(self):
        return '<File %s>' % self.id

    def print(self):
        """
        Prints class elements
        :parameter
        :return: --- <class 'NoneType'>
        """
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))

    def to_dict(self):
        return vars(self)

    def get_stability(self):
        return self.stability

    def get_prevalence(self):
        return self.prevalence

    def get_impact(self):
        return self.impact


# ================

