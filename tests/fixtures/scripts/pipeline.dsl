# Internal test pipeline for file-based execution
load dataset from "regression.csv"
preprocess dataset using target
train model using linear_regression on dataset
evaluate model using metrics
summary model
