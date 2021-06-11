#!/usr/bin/python

import matplotlib.pyplot as plt
from ml_helpers import evaluate_model

from svm_author_id import best_svc
from dt_author_id import best_dt
from adaboost import best_adaboost
from random_forest import best_random_forest
from sklearn.naive_bayes import GaussianNB



top_accuracy = 0
best_model = {}

# Gaussian Naive Bayes
nb_model = GaussianNB()
print()
print("GAUSSIAN NAIVE BAYES CLASSIFIER")
print()
training_time, predict_time, nb_accuracy = evaluate_model(nb_model, features_train, features_test, labels_train, labels_test)
print(f"Training time: {round(training_time, 3)}s")
print(f"Prediction time: {round(predict_time, 3)}s")
print(f"Accuracy: {nb_accuracy}")

# Support Vector Machine
kernels = ["linear", "rbf"]
Cs = [0.01, 0.1, 1, 10, 100, 1000, 10000]
gammas = ["scale", "auto", 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
svc, svc_params = best_svc(kernels, Cs, gammas, features_train, features_test, labels_train, labels_test)

# Decision Tree
min_splits = [2, 10, 20, 40, 50, 100]
random_state = 25
dt, dt_params = best_dt(min_splits, random_state, features_train, features_test, labels_train, labels_test)

# Random Forest
n_estimators_list = [5, 10, 20, 50, 80, 100, 200, 500, 1000]
min_samples_split = [2, 10, 20, 40, 50, 100]
random_state = 25
rand_forest, rand_forest_params = best_random_forest(n_estimators_list, min_samples_split, random_state, features_train, features_test, labels_train, labels_test)

# AdaBoost
n_estimators_list = [5, 10, 20, 50, 80, 100, 200, 500, 1000]
learning_rates = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
random_state = 25
adaboost, adaboost_params = best_adaboost(n_estimators_list, learning_rates, random_state, features_train, features_test, labels_train, labels_test)

models = {  adaboost: adaboost_params,
            rand_forest: rand_forest_params,
            dt: dt_params,
            svc: svc_params,
            nb_model: {"accuracy": nb_accuracy}}

for model, parameters in models.items():
    if parameters["accuracy"] > top_accuracy:
        top_accuracy = parameters["accuracy"]
        best_model = {"model": model, "params": parameters}

print("Best model used: ")
print(f"Model = {best_model['model']}")
print(f"Parameters = {best_model['params']}")
training_time, predict_time, accuracy = evaluate_model(best_model["model"], features_train, features_test, labels_train, labels_test)
print("-------------------------------------------------")
print(f"Training time: {round(training_time, 3)}s")
print(f"Prediction time: {round(predict_time, 3)}s")
print(f"Accuracy: {accuracy}")
print()

clf = best_model["model"]