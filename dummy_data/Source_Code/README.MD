# Synthetic Data Generator - Source Code

In this folder, there are .py files that are orchestrated in an OOP way to build a sample engine, with the following short description: (each file has docstring documentation)

# Group.py
 Abstract class:
 - Group A: Set Files that have a very low nr of changes(5-10%) and have impact in a lot of tests(50-100%), but rarely cause test breaks;
 - Group B: Set of files that have a very low nr of changes (5-10%) and have impact in a small amount of tests,
        with randomized break rates;
 - Group C: Set of files with frequent change rate (50-100%) and impact small amount of tests(5-20%), with
        randomized break rates;
 - Group D: Set of files with random change rate and impact in random amount of tests, with random break rates.
 
  
# File.py
Inherits from Group Class - Represents files in a repository:
## Attributes
- id:
- prevalence:
- stability:
- impact:
  
# Test.py
Inherits from Group Class - Represents Tests in a repository:
## Attributes
- id:test name
- size: small, medium or large test
- time_to_run: time in minutes to run

# Commit.py
Represents one commit to the repository:
## Attributes:
- author: person who did the commit
- timestamp: time of commit
- modified_files: files to commit, based on prevalence
- broken_files: files that are broken, based on stability

# SampleGenerator.py
Given input of problem, generates list of files, tests and commits (builds the repository)
## Attributes:
- nr_files: number of files in database
- nr_tests: number of tests that can be applied
- nr_commits: number of commits made
- nr_developers: team size

# DataFrame.py
Output class for documents.
## Attributes
- S: SampleGenerator class element

# Tools.py
Auxiliary class for some tools regarding random numbers and timestamps
