#!/usr/bin/python

from sklearn.naive_bayes import GaussianNB
from ml_helpers import evaluate_model

# Accuracy of 97.3%
print()
print("GAUSSIAN NAIVE BAYES CLASSIFIER")
print()
nb_model = GaussianNB()
training_time, predict_time, accuracy = evaluate_model(nb_model, features_train, features_test, labels_train, labels_test)
print(f"Training time: {round(training_time, 3)}s")
print(f"Prediction time: {round(predict_time, 3)}s")
print(f"Accuracy: {accuracy}")
