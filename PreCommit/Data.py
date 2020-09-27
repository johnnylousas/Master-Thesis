import itertools
from abc import abstractmethod

import numpy as np
import pandas as pd

from Visualizer import Visualizer


class Data:

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def get_data_info(self):
        pass


class DataCI(Data, Visualizer):
    """
    Receives, transforms, analyzes and cleans raw data.
     into pairs of (files, tests), by iterating through every
    revision and combining every single file with every single test.
    """

    def __init__(self, commits, test_details, test_history, mod_files, start_date: str = '2019-03-11 00:00:00.000000'):
        """
        Data Class Constructor
        :param commits:
        :param test_details:
        :param test_history:
        :param mod_files:
        """
        self.commits = commits
        self.test_details = test_details
        self.test_history = test_history
        self.mod_files = mod_files

        self.transform()
        self.clean_files(start_date=start_date)
        self.df_link = self.create_data_input(remove_flaky=True)
        self.clean_tests()

        # count number of distinct directories and files
        self.all_tests = list(self.df_link.name.explode().unique())
        self.all_files = list(self.df_link.mod_files.explode().unique())

        self.file_index = {file: idx for idx, file in enumerate(self.all_files)}
        self.index_file = {idx: file for file, idx in self.file_index.items()}

        self.test_index = {test: idx for idx, test in enumerate(self.all_tests)}
        self.index_test = {idx: test for test, idx in self.test_index.items()}
        print(f'There are {len(self.all_files)} unique files and {len(self.all_tests)}')

        # create pairs
        self.pairs = self.create_pairs()
        self.get_most_frequent_pairs()

    def transform(self):
        """
        For commit list, removes missing values and converts timestamp to datetime
        For test details, replaces None value durations by 2 minutes. (average duration)
        For test history, drops NaN values.
        For modified files, splits string into list of strings and cleans file path to format dir/filename
        :return:
        """
        # commit list
        self.commits['changes'] = self.commits['changes'].map(lambda x: x.lstrip('#').rstrip('aAbBcC'))
        self.commits[['nbModifiedFiles', 'nbAddedFiles']] = self.commits['changes'].str.split('+', expand=True)
        self.commits[['nbModifiedFiles', 'nbRemovedFiles']] = self.commits['nbModifiedFiles'].str.split('-',
                                                                                                        expand=True)
        self.commits[['nbAddedFiles', 'nbRemovedFiles']] = self.commits['nbAddedFiles'].str.split('-', expand=True)
        self.commits[['nbModifiedFiles', 'nbMovedFiles']] = self.commits['nbModifiedFiles'].str.split('~', expand=True)
        self.commits[['nbRemovedFiles', 'nbMovedFiles']] = self.commits['nbRemovedFiles'].str.split('~', expand=True)

        # Missing and Empty Values
        self.commits.nbModifiedFiles.fillna(value='0', inplace=True)
        self.commits.nbModifiedFiles.replace('', '0', inplace=True)
        self.commits.nbAddedFiles.fillna(value=0, inplace=True)
        self.commits.nbAddedFiles.replace('', '0', inplace=True)
        self.commits.nbRemovedFiles.fillna(value=0, inplace=True)
        self.commits.nbRemovedFiles.replace('', '0', inplace=True)
        self.commits.comment.fillna(value='no_comment', inplace=True)
        self.commits.nbMovedFiles.fillna(value=0, inplace=True)
        self.commits.nbMovedFiles.replace('', '0', inplace=True)

        self.commits.timestamp = pd.to_datetime(self.commits.timestamp, format='%Y-%m-%d %H:%M:%S')
        self.commits = self.commits[['rev', 'user', 'timestamp', 'nbModifiedFiles', 'nbAddedFiles',
                                     'nbRemovedFiles', 'nbMovedFiles', 'comment']]

        # test details list - replace None test duration
        self.test_details.duration = self.test_details.duration.replace(to_replace='None', value=2)
        self.test_details['duration'] = pd.to_datetime(self.test_details['duration']).sub(
            pd.Timestamp('00:02:00')).dt.seconds
        self.test_details['duration'] = np.round(self.test_details['duration'] / 60, 3)
        self.test_details['name'] = self.test_details['name'].str.replace(',',
                                                                          ';')  # replace every comma in test name for char

        # Test history list - remove revisions without label
        i = self.test_history[(self.test_history.revision == 'None')].index
        self.test_history = self.test_history.drop(i)

        # transform modified files list
        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda x:
                                                                        x.strip("[]").replace("'", "").split(", "))

        def get_filename(column):
            import os
            li = []
            for i in column:
                file = os.path.basename(i)
                path = os.path.normpath(i)
                if len(path.split(os.sep)) > 1:
                    li.append(path.split(os.sep)[-2] + "/" + file)
                else:
                    li.append(str(path.split(os.sep[0])) + "/" + file)
            return li

        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(
            lambda x: get_filename(x))  # remove full path of file
        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(
            lambda x: list(pd.unique(x)))  # remove duplicate files on each commit

    def get_data_info(self):
        pass

    def clean_files(self, start_date: str = '2019-03-11 00:00:00.000000'):
        """
        Some files present in the data are deprecated or unused. Thus we only want to keep relevant files.
        To do that, only files that have been modified in the past year are stored, otherwise it gets removed.
        :param start_date: Date threshold to remove modified files
        :return:
        """
        print(f'Removing files that are not modified since {start_date}')
        all_files = list(self.mod_files['mod_files'].explode().unique())
        print(f'There are {len(all_files)} files before cleaning')
        d = {k: v for v, k in enumerate(all_files)}
        index_file = {idx: file for file, idx in d.items()}

        def encode_files(m: list):
            idx = [d[k] for k in m]  # match indexes of tests in current revision
            return list(set(idx))

        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda x:
                                                                        encode_files(x))

        self.mod_files['timestamp'] = pd.to_datetime(self.mod_files['timestamp'])
        dt = pd.to_datetime(start_date)
        start = self.mod_files['timestamp'].sub(dt).abs().idxmin()

        file_list = list(range(0, len(all_files)))

        # get list of files not modified since 2019
        for i in self.mod_files.iloc[start:]['mod_files']:
            for f in file_list:
                if f in i:
                    file_list.remove(f)

        def remove_old_files(t: list):
            l1 = [x for x in t if x not in file_list]
            if l1:
                return l1
            else:
                return None

        # remove non relevant files from mod_files column
        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda t: remove_old_files(t))
        self.mod_files.dropna(inplace=True)  # Drop None values

        def recover_files(m: list):
            idx = [index_file[k] for k in m]  # match indexes of tests in current revision
            return idx

        self.mod_files['mod_files'] = self.mod_files['mod_files'].apply(lambda t: recover_files(t))
        print(f"There are {len(list(self.mod_files['mod_files'].explode().unique()))} files after cleaning \n")
        print(
            f'Percentage of files removed {np.round(100 * (1 - len(list(self.mod_files["mod_files"].explode().unique())) / len(all_files)), 2)} %')

    def create_data_input(self, remove_flaky: bool = False):
        """
        Creates unified input data for ML algorithm, where columns are (revision, mod_files and test names)
        :return: df
        """
        self.test_history = self.test_history.merge(self.test_details[['name', 'id']], how='inner', left_on='test_id',
                                                    right_on='id')
        # self.test_history.drop_duplicates(keep='first', inplace=True)   # drop rows where tests are applied more than
        # once and have the same result

        print(f'\nNumber of tests - {len(list(self.test_history["name"].explode().unique()))}')

        if remove_flaky:
            self.test_history.drop_duplicates(subset=['revision', 'name'], keep=False, inplace=True)  # remove flaky
            # tests
            print(f'Number of non-flaky tests - {len(list(self.test_history["name"].explode().unique()))}')

        self.test_history = self.test_history.groupby(['revision'])['name'].apply(', '.join).reset_index()  # by name

        self.test_history['revision'] = self.test_history['revision'].astype(int)
        self.test_history = self.test_history.sort_values(by=['revision'])
        self.test_history = self.test_history.reset_index()
        self.test_history.drop(columns=['index'], inplace=True)

        self.test_history['name'] = self.test_history['name'].apply(lambda x: x.split(', '))
        self.test_history['name'] = self.test_history['name'].apply(lambda x: list(pd.unique(x)))

        df = self.test_history.merge(self.mod_files, how='inner', on='revision')
        return df[['revision', 'mod_files', 'name']]

    def clean_tests(self, threshold: int = 10):
        """
        Drops tests from data that cause less than threshold transitions, i.e. very stable tests.
        """
        import collections

        # Count test frequency
        dist = self.df_link.name.explode().values
        dist = collections.Counter(dist)

        # Threshold
        good_tests = [k for k, v in dist.items() if float(v) >= threshold]

        print("** Data Cleaning - Removing Stable Tests**")
        print(f'   Number of tests -> {len(list(self.df_link["name"].explode().unique()))}')
        print(f'   Threshold -> {10}')
        print(f'   Percentage of tests above threshold - {np.round(100 * len(good_tests) / len(dist.keys()), 2)}')
        print(f'   Total number of transitions - {sum(dist.values())}')
        print(f'   Average number of transitions per test - {sum(dist.values()) / len(dist.keys())}')

        # Remove test below threshold from data
        def remove_stable_tests(t: list):
            l1 = [x for x in t if x in good_tests]
            if l1:
                return l1
            else:
                return None

        print(f'   Data shape - {self.df_link.shape}')
        self.df_link['name'] = self.df_link['name'].apply(lambda t: remove_stable_tests(t))

        # Drop None values
        self.df_link.dropna(inplace=True)
        print(f'\n   Number of tests after cleaning-> {len(list(self.df_link["name"].explode().unique()))}')
        print(f'   Data shape - {self.df_link.shape} - after cleaning')

    def create_pairs(self):
        pairs = []
        for row in self.df_link.iterrows():
            pairs.extend(list(itertools.product(row[1]['mod_files'], row[1]['name'])))
        pairs = list(map(lambda t: (self.file_index[t[0]], self.test_index[t[1]]), pairs))

        print(f'there are {len(set(pairs))} pairs')
        return pairs

    def get_most_frequent_pairs(self):
        # Most often pairs
        from collections import Counter
        x = Counter(self.pairs)
        print(f'Most often pairs {sorted(x.items(), key=lambda x: x[1], reverse=True)[:5]}')
        print(f'Nr of pairs - {len(self.pairs)}')
        print(f'Nr of pairs after threshold {len([k for k, v in x.items() if float(v) > 1])}')
        self.pairs = [k for k, v in x.items() if float(v) >= 2]

    def to_csv(self):
        self.df_link.to_csv('../pub_data/df.csv')
