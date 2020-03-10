import random
import pandas as pd
from faker import Faker
import tools
import datetime

# Commit Class
# ================
from dummy_data import File


class Commit:

    def __init__(self, author: str, timestamp: str, files: list):
        """
        Emulates a commit to the repository.
        :parameter  commit_id: unique identifier
                    author: person who did the commit
                    timestamp:
                    modified_files:
                    broken_files
        :return: --- <class 'NoneType'>
        """
        fake = Faker()
        Faker.seed(2)

        self.commit_id = fake.ean8()
        self.author = author
        self.timestamp = timestamp
        self.modified_files = files
        self.broken_files = self.breakage()

    def breakage(self):
        """
        Checks which of the modified files are broken
        :parameter
        :return: --- <class 'list'>
        """
        broken_files = []
        for i in self.modified_files:
            chance = random.randint(0, 100)
            stability = i.get_stability()
            if chance > stability:
                broken_files.append(i)
        return broken_files

    def get_broken_files(self):
        """
        return list of broken files
        :parameter
        :return: --- <class 'list'>
        """
        return self.broken_files

    def to_dict(self):
        return vars(self)

    def print(self):
        """
        Prints class attributes
        :parameter
        :return: --- <class 'list'>
        """
        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))

    def __repr__(self):
        return "<Commit commit_id: %s author: %s timestamp: %s modified_files:%s broken_files:%s>" % (self.commit_id,
                                                                                                   self.author,
                                                                                                   self.timestamp,
                                                                                                   self.modified_files,
                                                                                                   self.broken_files)

    def get_author(self):
        return self.author

    def get_timestamp(self):
        return self.timestamp

    def get_nr_modified_files(self):
        return len(self.modified_files)

# ================

