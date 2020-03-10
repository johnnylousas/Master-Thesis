import random
from faker import Faker


# Test Class
# ================


class Test:

    def __init__(self):
        """
        Initializes a Test case.
        Each test is split into groups (clusters).
        :parameter  id: filename
                    group: file group
                    size: small, medium or large test case
                    time_to_run: time a test takes to run, in minutes.
        :return: --- <class 'NoneType'>
        """
        random.seed()
        fake = Faker()

        groups = ['A', 'B', 'C', 'D']
        categories = ['small', 'medium', 'large']

        self.id = fake.file_name(category=None, extension='test')
        self.group = random.sample(groups, k=1)[0]
        self.size = random.choice(categories)

        self.time_to_run = self.run_time()

    def run_time(self):
        """
        Estimates time tests take to run, in minutes
        :parameter
        :return: --- <class 'int'>
        """
        if self.size.__eq__('small'):
            return random.randint(1, 5)
        elif self.size.__eq__('medium'):
            return random.randint(5, 15)
        elif self.size.__eq__('large'):
            return random.randint(15, 45)

    def print(self):
        """
        Prints class attributes
        :parameter
        :return: --- <class 'list'>
        """
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))

    def to_dict(self):
        """
        Returns class attributes as dictionary
        :parameter
        :return: --- <class 'dict'>
        """
        return vars(self)

    def __repr__(self):
        """
        Returns string corresponding to desired representation
        :parameter
        :return: --- <class 'string'>
        """
        return '<Test: %s>' % self.id

    def get_group(self):
        """
        Returns test group
        :parameter
        :return: --- <class 'string'>
        """
        return self.group

    def get_time(self):
        """
        Returns time to run
        :parameter
        :return: --- <class 'int'>
        """
        return self.time_to_run

# ================
