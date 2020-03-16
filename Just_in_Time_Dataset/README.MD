# JIT Dataset
From the paper "Deep Learning for Just-In-Time Defect Prediction" (2015), by Xinli Yang, David Lo, Xin Xia, Yun Zhang, and Jianling Sun,
we are introduced to the concept of change-level defect prediction, which enables developers to check and fix defects in time. (Similar to Kamei, et. al. (2013))
Their approach is to use deep learning algorithms to predict the likelihood a change has of causing a bug.

Just in time gathers datasets from six large open source projects, i.e., Bugzilla, Columba, JDT, Platform, Mozilla, and PostgreSQL, containing a total of 137,417 changes. 

This dataset contains the following features: 

- ns: The number of modified subsystems
- nd: The number of modified directories
- nf: The number of modified files
- entropy: Distribution of modified code across each file
- la: Lines of code added
- ld: Lines of code deleted
- lt: Lines of code in a file before the change
- fix: Whether or not the change is a defect fix
- ndev: The number of developers that changed the modified files
- age: The average time interval between the last and the current change
- nuc: The number of unique changes to the modified files
- exp: Developer experience
- rexp: Recent developer experience
- sexp: Developer experience on a subsystem
- bug: 1 if buggy, 0 otherwise

In this folder, I apply my own models to this dataset, such as, Logistic Regression, Decision Trees and Artificial Neural Networks.

# Code 

## classifier.py
  This class provides the tools needed to apply Logistic Regression, Decision Trees and Artificial Neural Network models to the dataset. Methods from this class englobe data visualization - boxplots, histograms, scatterplots, correlation maps, etc. - data preprocessing - through the method preprocess(option: str) it is possible to apply several preprocessing techniques, implemented in the file below.
  
## preprocess.py
  Preprocessing techniques like: MinMaxScaler, StandardScaler, SMOTE, UnderSampling, Near Miss, PCA, LDA, Feature Selection, etc.

## plotFunctions.py
  Functions relative to data visualization.
