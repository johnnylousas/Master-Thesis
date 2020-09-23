# Pre-Commit Neural Networks Embeddings

## Class Files 

### Data 

Reads .csv files, transforms , analyzes and provides relevant statistics about the data. 

### DataGenerator

Inherits from keras.Sequential and generates pairs (file, test, label) where positive labels are 1 if the connection exist in the data and 0 otherwise.

### Model 

Abstract class to choose which model is going to learn the relationship between files and tests.

## Prioritizer

Child class that inherits from Model.py, implementing Neural Networks Embeddings. Run Prioritizer.py, which contains the main function of the framework.
