
This folder contains a controlled environment that mimics what happens in a real life environment.

# Main tables 

## Commit list. 
This table contains the information typically obtained from a source control log: a chronological list of all changes and related information.
- commit_id: unique commit identifier
- author: name of the author of the change
- timestamp: modification time and date
- modified files: list of files modified by the commit
- broken files (*hidden information*): list of files with regressions

## Test files list. 
This table contains the list of all tests.
- test_id: unique test identifier
- time to run: time needed to run the test (everything else being equal, faster tests are better)
- depends on (be obtained by code coverage tools, but usually *hidden information*): list of files this test depends on

## Commit list.
This table contains the information typically obtained from a source control log: a chronological list of all changes and related information.
- commit_id: unique commit identifier
- author: name of the author of the change
- timestamp: modification time and date
- modified files: list of files modified by the commit
- broken files (hidden information): list of files with regressions

 NOTE: these attributes are set to be chosen randomly.

---

# Test strategy 

A naive approach if to run all teas for each commit. This solution does not scale. 
As project and teams gets bigger, the number of commits per day grows linearly, the number of test to run also grows linearly. 
All in all, with naive approach, the computing resources grows quadratically with the size of the project.

A more swift approach would be to only apply tests periodically, rather than checking every single commit. Then when faults are detected, we need binary search to trace back the initial faulty commit.

Also, people want to quickly know the status of the tests when they do a change (pre-commit hook, or commit watchdog).

The objective is, under given constraint of time and resources, to give the best Live estimate of the status of the project.
The performance of a given strategy can be measured, a posteriori by looking at the history of the test that have been run and the status reported.

## Test run history

This table contains the history of all tests that have been run and different approaches can be taken. The code allows the user to choose between generating this table using Naive or Accumulative approach.
- commit_id: what version of the project has been tested
- test_id: what test has been run 
- timestamp: when the test has been run
- test_status: status of this test
- this_commit_estimated_status: an estimation of the status of a project as of this commit (from what has been run so far) *innovation!*


A good test strategy would use heuristics to exploit common usage pattern. Machine learning can be used to learn those heuristics automatically.

Typical heuristics, related to for files patterns: 
-	some files are stable and affect a lot of tests (example : the standard libraries, the core functions, like LAPACK pakage, etc)
-	some files are moving much and affect few tests (for instance a new cool feature)
-	slso you have cross effects (modifying a database will have wide effect, so does core function, and there is overlap

Typical heuristics, related to user commit  pattern: 
-	when an author makes a change that caused a regression, his subsequent commits are likely to affect the same tests (to fix them or break them again).




- when an author makes a change that causes regression, his subsequent commits are likely to affect the same tests.
- some tests are good indicator of whether other tests will fail
- etc...


*When generating playground Commit list and Test files list, you should make sure you emulate those patterns. *


By default, the repository created in the code has 10 files, 10 tests, 10 commits and 4 developers in the team, to facilitate human interpretation. All of these parameters are free to be tuned. The output of the code are 4 .csv files corresponding to the 4 tables described above.


